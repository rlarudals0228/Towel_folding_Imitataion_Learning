#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realsense_towel_metrics.py (Decision version)

- RealSense depth → plane fit → residual map
- Residual & color mask → largest contour (towel)
- Metrics: RectFit, Height std, Height range
- Threshold 기반 decision: fold / flatten
- Overlay: metrics + decision 글씨 표시 (green)
- Topics:
    /towel/overlay   (sensor_msgs/Image)
    /towel/metrics   (std_msgs/String, JSON)
    /towel/decision  (std_msgs/String, "fold"/"flatten")
"""

import argparse, json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy as RP, QoSHistoryPolicy as HP, QoSDurabilityPolicy as DP
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge


# ---------------- QoS ----------------
SENSOR_QOS = QoSProfile(depth=10, reliability=RP.BEST_EFFORT, durability=DP.VOLATILE, history=HP.KEEP_LAST)
INFO_QOS   = QoSProfile(depth=10, reliability=RP.RELIABLE,    durability=DP.VOLATILE, history=HP.KEEP_LAST)

# ---------------- 3D & Plane ----------------
def deproject_points(depth_m, K, step=4):
    h, w = depth_m.shape
    fx, fy, cx, cy = K
    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    z = depth_m[gy, gx]
    valid = z > 0.05
    if not np.any(valid):
        return np.empty((0, 3), np.float32)
    u = gx[valid].astype(np.float32)
    v = gy[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x, y, z], axis=1)

def fit_plane_ransac(P, thr=0.003, iters=200):
    if P.shape[0] < 50:
        return None, None
    rng = np.random.default_rng(0)
    N = P.shape[0]
    best = None
    best_cnt = -1
    for _ in range(iters):
        i = rng.choice(N, 3, replace=False)
        p1, p2, p3 = P[i]
        n = np.cross(p2 - p1, p3 - p1)
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n /= L
        d = -np.dot(n, p1)
        cnt = np.sum(np.abs(P @ n + d) < thr)
        if cnt > best_cnt:
            best = (n, d)
            best_cnt = cnt
    if best is None:
        return None, None
    inl = np.abs(P @ best[0] + best[1]) < thr
    Q = P[inl]
    if Q.shape[0] < 3:
        return best
    C = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - C, full_matrices=False)
    n = Vt[-1]
    n /= (np.linalg.norm(n) + 1e-9)
    d = -np.dot(n, C)
    return n, d

def plane_residual_map(depth_m, K, n, d):
    h, w = depth_m.shape
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    Z = depth_m
    valid = Z > 0.05
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    R = np.zeros_like(Z, np.float32)
    P = np.dstack([X, Y, Z])
    R[valid] = np.abs(P[valid] @ n + d)
    return R

# ---------------- Masks & Metrics ----------------
def towel_mask_from_residual(res_m, lo_mm=1.5, hi_mm=50.0):
    mm = res_m * 1000.0
    m = ((mm > lo_mm) & (mm < hi_mm)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    return m

def bg_color_mask_bylab(color_bgr):
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]; b = lab[:, :, 2]
    bor = 10
    border_a = np.concatenate([a[:bor,:].ravel(), a[-bor:,:].ravel(), a[:, :bor].ravel(), a[:, -bor:].ravel()])
    border_b = np.concatenate([b[:bor,:].ravel(), b[-bor:,:].ravel(), b[:, :bor].ravel(), b[:, -bor:].ravel()])
    a0 = np.median(border_a); b0 = np.median(border_b)
    dist = np.sqrt((a - a0)**2 + (b - b0)**2).astype(np.float32)
    dist_u8 = np.clip(dist * 2.0, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m

def _remove_border_connected(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    out = m.copy()
    visited = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if out[0, x]:      cv2.floodFill(out, visited, (x, 0),   0)
        if out[h-1, x]:    cv2.floodFill(out, visited, (x, h-1), 0)
    for y in range(h):
        if out[y, 0]:      cv2.floodFill(out, visited, (0, y),   0)
        if out[y, w-1]:    cv2.floodFill(out, visited, (w-1, y), 0)
    return out

def outer_towel_mask(color_bgr, mask_residual):
    m_col = bg_color_mask_bylab(color_bgr)
    m = cv2.bitwise_and(mask_residual, m_col)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k2, iterations=1)
    m = _remove_border_connected(m)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(m), None
    c = max(cnts, key=cv2.contourArea)
    mask_largest = np.zeros_like(m)
    cv2.drawContours(mask_largest, [c], -1, 255, thickness=cv2.FILLED)
    return mask_largest, c

def compute_height_stats(res_m, mask):
    v = res_m[mask > 0] * 1000.0  # mm
    if v.size == 0:
        return dict(mean=0.0, std=0.0, range_mm=0.0)
    p05 = float(np.percentile(v, 5.0))
    p95 = float(np.percentile(v, 95.0))
    return dict(
        mean=float(v.mean()),
        std=float(v.std()),
        range_mm=float(p95 - p05),
    )

def compute_rect_metrics(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, None, None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    rect_area = float(max(w, 1.0) * max(h, 1.0))
    box = cv2.boxPoints(rect).astype(np.int32)
    rm = np.zeros_like(mask)
    cv2.fillPoly(rm, [box], 255)
    mask_bin = (mask > 0)
    rect_bin = (rm > 0)
    inter = np.logical_and(mask_bin, rect_bin).sum()
    uni   = np.logical_or(mask_bin, rect_bin).sum()
    iou  = inter / float(uni) if uni else 0.0
    return float(iou), float(rect_area), c, box

# ---------------- ROS Node ----------------
class NodeTowel(Node):
    def __init__(self, a):
        super().__init__('towel_metrics_node')
        self.bridge = CvBridge()
        self.decimate = a.decimate
        self.res_lo = a.res_lo_mm
        self.res_hi = a.res_hi_mm
        self.min_area_ratio = a.min_area_ratio
        self.max_area_ratio = a.max_area_ratio
        self.metrics_topic = a.metrics_topic
        self.overlay_topic = a.overlay_topic

        self.rect_thr = a.rect_thr
        self.std_thr_mm = a.std_thr_mm
        self.range_thr_mm = a.range_thr_mm

        self.sub_color = self.create_subscription(Image, a.color, self.cb_color, SENSOR_QOS)
        self.sub_depth = self.create_subscription(Image, a.depth, self.cb_depth, SENSOR_QOS)
        self.sub_info  = self.create_subscription(CameraInfo, a.info,  self.cb_info,  INFO_QOS)

        self.pub_metrics  = self.create_publisher(String, self.metrics_topic, 10)
        self.pub_overlay  = self.create_publisher(Image,  self.overlay_topic, 10)
        self.pub_decision = self.create_publisher(String, "/towel/decision", 10)
        self.current_state = "unknown"

        self.K = None
        self.last_color = None
        self.last_decision = None
        self.GREEN = (0, 255, 0)

    def cb_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def cb_color(self, msg: Image):
        try:
            self.last_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #self.get_logger().info(f"color frame received: encoding={msg.encoding}, shape={self.last_color.shape}")
        except Exception as e:
            self.get_logger().warn(f'color cv_bridge: {e}')
            self.last_color = None

    def cb_depth(self, msg: Image):
        if self.K is None or self.last_color is None:
            return
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough').astype(np.float32) * 0.001
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough').astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'depth cv_bridge: {e}')
            return

        
        
        color = self.last_color.copy()
        fx, fy, cx, cy = self.K
        K = (fx, fy, cx, cy)

        if self.decimate > 1:
            h, w = depth.shape
            depth = cv2.resize(depth, (w // self.decimate, h // self.decimate), interpolation=cv2.INTER_NEAREST)
            color = cv2.resize(color, (w // self.decimate, h // self.decimate), interpolation=cv2.INTER_LINEAR)
            K = (fx / self.decimate, fy / self.decimate, cx / self.decimate, cy / self.decimate)

        P = deproject_points(depth, K, step=4)
        n, d = fit_plane_ransac(P, thr=0.003, iters=200)
        if n is None:
            return
        res = plane_residual_map(depth, K, n, d)

        mask_res = towel_mask_from_residual(res, self.res_lo, self.res_hi)
        mask, contour = outer_towel_mask(color, mask_res)

        H, W = mask.shape
        frame_area = float(H * W)
        area_px = float(cv2.contourArea(contour)) if contour is not None else 0.0
        detected = bool(contour is not None and
                        (self.min_area_ratio * frame_area <= area_px <= self.max_area_ratio * frame_area))

        rect_iou = rect_area = 0.0
        box = None
        hs = dict(mean=0.0, std=0.0, range_mm=0.0)

        if detected:
            hs = compute_height_stats(res, mask)
            rect_iou, rect_area, _, box = compute_rect_metrics(mask)

        # --- decision logic ---
        #decision = "fold"
        decision = "flatten"
        if detected and rect_iou >= self.rect_thr and hs["std"] <= self.std_thr_mm and hs["range_mm"] <= self.range_thr_mm:
           decision = "fold"
        
        self.pub_decision.publish(String(data=decision))
        #self.get_logger().info(f"Decision: {decision}")

        if decision != self.last_decision:
            self.pub_decision.publish(String(data=decision))
            self.get_logger().info(f"Decision changed: {decision}")
            self.last_decision = decision
            
        # 현재 상태 저장
        self.current_state = decision
        # overlay
        overlay = color
        if detected and contour is not None:
            cv2.drawContours(overlay, [contour], -1, self.GREEN, 2)
            if box is not None:
                cv2.polylines(overlay, [box], True, self.GREEN, 2)
        else:
            cv2.putText(overlay, "NO TOWEL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(overlay, "NO TOWEL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, self.GREEN, 2, cv2.LINE_AA)

        # 글자 출력
        y = 28
        def put_line(txt, step=26):
            nonlocal y
            y += step
            cv2.putText(overlay, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(overlay, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.GREEN, 2, cv2.LINE_AA)

        if detected:
            put_line(f"RectFit: {rect_iou*100:.1f}% (thr {self.rect_thr*100:.1f}%)")
            put_line(f"Height std: {hs['std']:.2f} mm (thr {self.std_thr_mm})")
            put_line(f"Height range: {hs['range_mm']:.2f} mm (thr {self.range_thr_mm})")
        else:
            put_line("Mask/area gating failed.")

        put_line(f"DECISION: {decision.upper()}", step=32)

        img = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        img.header = msg.header
        self.pub_overlay.publish(img)

        # publish metrics + decision
        out = {
            "detected": detected,
            "frame_area_px": frame_area,
            "mask_area_px": area_px,
            "rect_area_px": rect_area,
            "rect_iou": rect_iou,
            "height_std_mm":   hs["std"],
            "height_range_mm": hs["range_mm"],
            "height_mean_mm":  hs["mean"],
            "decision": decision
        }
        self.pub_metrics.publish(String(data=json.dumps(out, ensure_ascii=False)))
        self.current_state = decision

# ---------------- Entrypoint ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--color', default='/camera/external_camera/color/image_rect_raw')
    p.add_argument('--depth', default='/camera/external_camera/depth/image_rect_raw')
    p.add_argument('--info',  default='/camera/external_camera/depth/camera_info')
    p.add_argument('--decimate', type=int, default=1)
    p.add_argument('--res-lo-mm', dest='res_lo_mm', type=float, default=1.5)
    p.add_argument('--res-hi-mm', dest='res_hi_mm', type=float, default=50.0)
    p.add_argument('--min-area-ratio', type=float, default=0.02)
    p.add_argument('--max-area-ratio', type=float, default=0.55)
    p.add_argument('--metrics-topic', default='/towel/metrics')
    p.add_argument('--overlay-topic', default='/towel/overlay')
    p.add_argument('--rect-thr', type=float, default=0.85)
    p.add_argument('--std-thr-mm', type=float, default=7.0)
    p.add_argument('--range-thr-mm', type=float, default=18.0)
    
    args,_ = p.parse_known_args()
    return args

def main():
    args = parse_args()
    rclpy.init()
    n = NodeTowel(args)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            n.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()
