# geometry_optimizer.py
import numpy as np
import cv2
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # ch = logging.StreamHandler() # 필요시 주석 해제
    # ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # logger.addHandler(ch)

class GeometryOptimizer:
    def __init__(self, snap_threshold_pixels=5, min_vertices_for_polygon=3,
                 min_edge_length_pixels=1):
        self.snap_threshold = float(snap_threshold_pixels)
        self.min_vertices = int(min_vertices_for_polygon)
        self.min_edge_length = float(min_edge_length_pixels)
        logger.info(f"GeometryOptimizer 초기화 (기본 최적화 모드): 스냅={self.snap_threshold}px, "
                    f"최소점={self.min_vertices}, 최소선길이={self.min_edge_length}px")

    def optimize_room_geometries(self, room_contours_list,
                                 original_image_for_visualization=None,
                                 roi_offset=(0,0)): # roi_offset은 시각화 시 윤곽선 위치 조정을 위해 유지
        logger.info(f"기본 방 형상 최적화 시작 (스냅핑, 정리). 입력 방 개수: {len(room_contours_list)}")

        if not room_contours_list:
            vis_img = self._get_empty_visualization_image(original_image_for_visualization)
            logger.warning("입력된 방 윤곽선 리스트가 비어있습니다.")
            return [], vis_img

        all_vertices_info = self._collect_all_vertices(room_contours_list)
        grouped_vertices = self._group_snap_vertices(all_vertices_info)
        self._update_vertices_with_group_centroids(all_vertices_info, grouped_vertices)
        
        # 기본 최적화된 윤곽선 (스냅핑, 정리, 유효성 검사)
        optimized_contours = self._reconstruct_and_validate_contours(all_vertices_info, room_contours_list)
        logger.info(f"기본 최적화 완료. 최종 {len(optimized_contours)}개의 유효 윤곽선 생성.")

        visualization_image = None
        if original_image_for_visualization is not None:
            visualization_image = self._create_visualization_image(
                original_image_for_visualization,
                room_contours_list,         # 최초 입력된 원본 윤곽선
                optimized_contours,   # 기본 최적화가 끝난 최종 윤곽선
                roi_offset
            )

        return optimized_contours, visualization_image

    def _get_empty_visualization_image(self, base_image):
        if base_image is None: return None
        return cv2.cvtColor(base_image.copy(), cv2.COLOR_GRAY2BGR) if len(base_image.shape) == 2 or base_image.shape[2] == 1 else base_image.copy()

    def _collect_all_vertices(self, room_contours_list):
        all_vertices = []
        for r_idx, contour in enumerate(room_contours_list):
            if not isinstance(contour, np.ndarray) or contour.ndim < 2: continue
            for v_idx, point_wrapper in enumerate(contour):
                if not (isinstance(point_wrapper, (list, np.ndarray)) and len(point_wrapper) > 0 and
                        isinstance(point_wrapper[0], (list, np.ndarray)) and len(point_wrapper[0]) == 2): continue
                all_vertices.append({'x_orig': float(point_wrapper[0][0]), 'y_orig': float(point_wrapper[0][1]),
                                     'x_new': float(point_wrapper[0][0]), 'y_new': float(point_wrapper[0][1]),
                                     'room_idx': r_idx, 'v_idx_in_room': v_idx, 'snap_group_id': -1,
                                     'global_id': len(all_vertices)})
        return all_vertices

    def _group_snap_vertices(self, all_vertices_info):
        parent = {v['global_id']: v['global_id'] for v in all_vertices_info}
        def find_set(vert_id):
            if parent[vert_id] == vert_id: return vert_id
            parent[vert_id] = find_set(parent[vert_id]); return parent[vert_id]
        def unite_sets(id1, id2):
            id1_root, id2_root = find_set(id1), find_set(id2)
            if id1_root != id2_root: parent[id2_root] = id1_root
        for i in range(len(all_vertices_info)):
            for j in range(i + 1, len(all_vertices_info)):
                v_i, v_j = all_vertices_info[i], all_vertices_info[j]
                if find_set(v_i['global_id']) == find_set(v_j['global_id']): continue
                if ((v_i['x_orig'] - v_j['x_orig'])**2 + (v_i['y_orig'] - v_j['y_orig'])**2) < self.snap_threshold**2:
                    unite_sets(v_i['global_id'], v_j['global_id'])
        final_groups = defaultdict(list)
        for v_info in all_vertices_info:
            root_id = find_set(v_info['global_id'])
            v_info['snap_group_id'] = root_id; final_groups[root_id].append(v_info['global_id'])
        return final_groups

    def _update_vertices_with_group_centroids(self, all_vertices_info, grouped_vertices):
        vertex_map = {v['global_id']: v for v in all_vertices_info}
        for group_id, member_ids in grouped_vertices.items():
            if not member_ids: continue
            sum_x = sum(vertex_map[mid]['x_orig'] for mid in member_ids)
            sum_y = sum(vertex_map[mid]['y_orig'] for mid in member_ids)
            centroid_x, centroid_y = sum_x / len(member_ids), sum_y / len(member_ids)
            for mid in member_ids: vertex_map[mid]['x_new'], vertex_map[mid]['y_new'] = centroid_x, centroid_y

    def _reconstruct_and_validate_contours(self, all_vertices_info, original_room_contours_list):
        temp_reconstructed_rooms = defaultdict(list)
        for v_info in all_vertices_info: temp_reconstructed_rooms[v_info['room_idx']].append(v_info)
        final_valid_contours = []
        for room_idx in range(len(original_room_contours_list)):
            if room_idx not in temp_reconstructed_rooms:
                final_valid_contours.append(np.array([], dtype=np.int32)); continue
            sorted_vertices = sorted(temp_reconstructed_rooms[room_idx], key=lambda v: v['v_idx_in_room'])
            points_float = np.array([[v['x_new'], v['y_new']] for v in sorted_vertices], dtype=np.float32)
            if points_float.shape[0] == 0:
                final_valid_contours.append(np.array([], dtype=np.int32)); continue
            cleaned_points = self._clean_polygon_edges(points_float)
            if len(cleaned_points) < self.min_vertices:
                final_valid_contours.append(np.array([], dtype=np.int32)); continue
            final_points_int = np.array([[[int(round(p[0])), int(round(p[1]))]] for p in cleaned_points], dtype=np.int32)
            final_valid_contours.append(final_points_int)
        return final_valid_contours

    def _clean_polygon_edges(self, points_float):
        if points_float.shape[0] < 2: return points_float
        cleaned = [points_float[0]]
        for i in range(1, len(points_float)):
            p_curr, p_prev = points_float[i], cleaned[-1]
            if np.allclose(p_curr, p_prev, atol=1e-3): continue
            if ((p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2) < self.min_edge_length**2: continue
            cleaned.append(p_curr)
        if len(cleaned) > 1 and np.allclose(cleaned[-1], cleaned[0], atol=1e-3): cleaned.pop()
        return np.array(cleaned, dtype=np.float32)

    def _create_visualization_image(self, base_image, original_contours, optimized_contours, offset=(0,0)):
        vis_img = self._get_empty_visualization_image(base_image)
        if vis_img is None: return None
        color_original, th_orig = (0, 255, 0), 1
        color_optimized, th_opt = (255, 0, 0), 3
        color_opt_verts, r_vert = (0, 0, 255), 3
        def apply_offset(cnt, off):
            return cnt + np.array([off[0], off[1]]).reshape(-1, 1, 2) if cnt is not None and len(cnt) > 0 else None
        for contour in original_contours:
            cnt_draw = apply_offset(contour, offset)
            if cnt_draw is not None: cv2.drawContours(vis_img, [cnt_draw], -1, color_original, th_orig, cv2.LINE_AA)
        for contour in optimized_contours:
            cnt_draw = apply_offset(contour, offset)
            if cnt_draw is not None and len(cnt_draw) > 0:
                cv2.drawContours(vis_img, [cnt_draw], -1, color_optimized, th_opt, cv2.LINE_AA)
                for pt_wrap in cnt_draw: cv2.circle(vis_img, tuple(pt_wrap[0]), r_vert, color_opt_verts, -1)
        return vis_img