# functional.py
import os
import cv2
import numpy as np
import ezdxf 
import logging

class ContourExtractor:
    """
    Class for extracting building outlines from preprocessed floor plan images.
    """

    def __init__(self):
        """Initialize the contour extractor"""
        pass

    def extract_contours(self, image, params):
        """
        Extract contours and their hierarchy from the preprocessed image.
        
        Args:
            image (numpy.ndarray): Preprocessed binary image. 
                                   Expected to be a 2D array (grayscale/binary).
            params (dict): Dictionary of contour extraction parameters.
                           (e.g., approx_epsilon, min_area_ratio)
                           
        Returns:
            tuple: (display_img, contours, hierarchy)
                   display_img: Image with all raw contours drawn (for debugging/visualization).
                   contours: List of all detected contours.
                   hierarchy: Contour hierarchy information from findContours.
        """
        if len(image.shape) > 2 and image.shape[2] == 3: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2: 
            gray = image.copy()
        else: 
            print(f"Unsupported image shape: {image.shape}")
            return np.zeros((100,100,3), dtype=np.uint8), [], None

        if np.max(gray) > 1 and len(np.unique(gray)) > 2: 
             _, binary = cv2.threshold(gray, params.get('threshold_value_for_contour', 127), 255, cv2.THRESH_BINARY)
        elif np.max(gray) == 1: 
            binary = gray * 255
        else: 
            binary = gray.copy()

        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return display_img, contours, hierarchy

    def filter_and_select_outline(self, image_shape, contours, hierarchy, params):
        """
        Filters contours to select the main building outline.
        Uses min_area_ratio relative to image_area.
        Removes raw contour logging.
        """
        if hierarchy is None or len(hierarchy) == 0:
            logging.warning("Hierarchy is missing or empty. Cannot select outline.")
            return [], None 
        
        actual_hierarchy = hierarchy
        if len(hierarchy.shape) == 3 and hierarchy.shape[0] == 1:
            actual_hierarchy = hierarchy[0]

        image_height, image_width = image_shape[:2]
        image_area = float(image_height * image_width) # 전체 이미지 면적 계산 (float으로)

        # Get parameters
        min_area_ratio = params.get('min_area_ratio', 0.01) 
        min_area_pixels_threshold = image_area * min_area_ratio 
        
        max_area_ratio_to_image = params.get('max_area_ratio_to_image', 0.8) 
        min_children_count = params.get('min_children_count', 20) 
        approx_epsilon_factor = params.get('approx_epsilon', 0.01) 
        min_vertices = params.get('min_vertices', 4)
        max_vertices = params.get('max_vertices', 300) 
        min_aspect_ratio = params.get('min_aspect_ratio', 0.05)
        max_aspect_ratio = params.get('max_aspect_ratio', 20.0)
        min_solidity = params.get('min_solidity', 0.65) 
        border_margin = params.get('border_margin', 25) # UI에서는 border_margin_filter
        filter_by_rectangle = params.get('filter_by_rectangle', False) # UI에서는 filter_by_rectangle_adv
        min_rect_ratio_thresh = params.get('min_rect_ratio', 0.3) # UI에서는 min_rect_ratio_adv
        max_rect_ratio_thresh = params.get('max_rect_ratio', 2.0) # UI에서는 max_rect_ratio_adv
        
        # These parameters are used in logging/conditions below, ensure they are defined or retrieved if needed
        strict_border_margin = params.get('strict_border_margin', 15) # 예시 값, UI에 없다면 기본값 사용 또는 params에서 가져오기
        min_area_ratio_for_border_check = params.get('min_area_ratio_for_border_check', 0.03)  # 예시 값
        max_building_candidate_area_ratio = params.get('max_building_candidate_area_ratio', 0.4) # 예시 값

        min_area_passed_details = [] 
        candidate_outlines = [] 

        logging.info(f"--- Contour Filtering Started (Total Contours: {len(contours)}) ---")
        logging.info(f"Image Area: {image_area:.0f} pixels.")
        logging.info(f"Applied Params: min_area_ratio={min_area_ratio} (Threshold: {min_area_pixels_threshold:.0f} pixels), "
                     f"max_area_ratio_to_image(border)={max_area_ratio_to_image}, min_children={min_children_count}, "
                     f"min_solidity={min_solidity}, approx_epsilon_factor={approx_epsilon_factor}, "
                     f"max_building_candidate_area_ratio={max_building_candidate_area_ratio}")

        # --- Raw Contour Coordinates & Area Logging 부분 제거됨 ---

        for i, contour in enumerate(contours): 
            area = cv2.contourArea(contour)
            
            if area < min_area_pixels_threshold:
                # 너무 작은 로그는 피하기 위해, 어느 정도 의미있는 크기인데 탈락한 경우만 로깅 (선택사항)
                if area > min_area_pixels_threshold * 0.05 and area > 50: # 예: 최소 기준의 5% 이상이고 50픽셀 이상
                     logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}]: Initial filter: Too small (Area < {min_area_pixels_threshold:.0f}).")
                continue
            
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            current_hierarchy_info = actual_hierarchy[i]
            parent_index = current_hierarchy_info[3]
            
            child_count = 0
            first_child_index_for_calc = current_hierarchy_info[2]
            if first_child_index_for_calc != -1:
                sibling_index_for_calc = first_child_index_for_calc
                while sibling_index_for_calc != -1:
                    child_count += 1
                    sibling_index_for_calc = actual_hierarchy[sibling_index_for_calc][0]
            
            perimeter = cv2.arcLength(contour, True)
            approx = contour 
            vertices_count = len(contour) 
            if perimeter > 0:
                epsilon = approx_epsilon_factor * perimeter 
                approx = cv2.approxPolyDP(contour, epsilon, True) 
                vertices_count = len(approx) 

            aspect_ratio = float(w_c) / h_c if h_c > 0 and w_c > 0 else 0
            
            solidity = 0
            if perimeter > 0: 
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
            
            rect_ratio_str = "N/A (Filter Off)"
            if filter_by_rectangle: # UI의 filter_by_rectangle_adv 와 연결
                if perimeter > 0: 
                    min_rect = cv2.minAreaRect(contour)
                    min_rect_rotated_area = min_rect[1][0] * min_rect[1][1]
                    if min_rect_rotated_area > 0:
                        rect_ratio_for_detail_val = area / min_rect_rotated_area
                        rect_ratio_str = f"{rect_ratio_for_detail_val:.2f}"
                    else:
                        rect_ratio_str = "N/A (Zero RotRect Area)"
                else:
                    rect_ratio_str = "N/A (Zero Perimeter)"

            details = {
                'id': i, 
                'contour_original': contour, 
                'contour_approx': approx,   
                'area': area, 
                'area_ratio_to_image': area / image_area if image_area > 0 else 0,
                'parent_id': parent_index, 
                'children': child_count,
                'solidity': solidity, 
                'vertices': vertices_count, 
                'bounding_rect': (x_c, y_c, w_c, h_c),
                'aspect_ratio': aspect_ratio,
                'rect_ratio_str': rect_ratio_str 
            }
            min_area_passed_details.append(details)
            
            is_touching_horizontal_borders = (y_c < strict_border_margin) or ((y_c + h_c) > (image_height - strict_border_margin))
            is_touching_vertical_borders = (x_c < strict_border_margin) or ((x_c + w_c) > (image_width - strict_border_margin))
            is_wide_and_touching = (w_c > image_width * 0.90) and is_touching_vertical_borders 
            is_tall_and_touching = (h_c > image_height * 0.90) and is_touching_horizontal_borders 

            if (area > image_area * min_area_ratio_for_border_check) and \
               (is_wide_and_touching or is_tall_and_touching):
                logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Bounds:({x_c},{y_c},{w_c},{h_c})]: Filtered by strict border check.")
                continue
            
            # UI의 max_area_ratio_to_image와 연결
            if area > image_area * max_area_ratio_to_image: 
                # UI의 border_margin_filter 와 연결 (여기서는 border_margin 사용)
                if x_c < border_margin and y_c < border_margin and \
                   (x_c + w_c) > (image_width - border_margin) and \
                   (y_c + h_c) > (image_height - border_margin):
                    logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}]: Filtered by secondary border check (for image border).")
                    continue
            
            if child_count < min_children_count:
                 logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Parent:{parent_index}, Children:{child_count}]: Filtered by min_children_count ({child_count} < {min_children_count}).")
                 continue
            logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Passed children_count filter.")

            if perimeter == 0: 
                logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Filtered: zero perimeter.")
                continue
            
            if not (min_vertices <= vertices_count <= max_vertices): 
                logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Filtered by vertices ({vertices_count}). Not in [{min_vertices},{max_vertices}].")
                continue
            
            if w_c == 0 or h_c == 0:
                logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Filtered: zero width/height bbox.")
                continue
            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Filtered by aspect_ratio ({aspect_ratio:.2f}). Not in [{min_aspect_ratio:.2f},{max_aspect_ratio:.2f}].")
                continue
            
            if solidity < min_solidity:
                logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Filtered by solidity ({solidity:.2f} < {min_solidity:.2f}).")
                continue
            
            if filter_by_rectangle: # UI의 filter_by_rectangle_adv
                current_rect_ratio = -1.0 
                try:
                    if rect_ratio_str.startswith("N/A"): 
                        pass 
                    else:
                        current_rect_ratio = float(rect_ratio_str)
                except ValueError:
                    logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Could not parse rect_ratio_str: {rect_ratio_str}. Skipping rect filter for this contour.")
                    current_rect_ratio = -1.0 # Skip filter if parse error
                
                if current_rect_ratio != -1.0 and not (min_rect_ratio_thresh <= current_rect_ratio <= max_rect_ratio_thresh):
                    logging.debug(f"  FilterLoop - Contour {i} [Area:{area:.0f}, Children:{child_count}]: Filtered by rect_ratio ({current_rect_ratio:.2f}). Not in [{min_rect_ratio_thresh:.2f},{max_rect_ratio_thresh:.2f}].")
                    continue

            logging.info(f"   >>> Contour {i} [Area:{area:.0f}, Parent:{parent_index}, Children:{child_count}, Solidity:{solidity:.2f}, Vertices:{vertices_count}] ADDED AS CANDIDATE. <<<")
            candidate_outlines.append(details) 

        logging.info(f"\n--- Detailed Info for {len(min_area_passed_details)} Contours that Passed Min Area Ratio ({min_area_ratio} => {min_area_pixels_threshold:.0f} px) Filter: ---")
        for detail_info in min_area_passed_details:
            logging.info(f"  MinAreaPassed - ID: {detail_info['id']}, Area: {detail_info['area']:.0f} (RatioToImg: {detail_info.get('area_ratio_to_image', 0):.4f}), Parent: {detail_info['parent_id']}, Children: {detail_info['children']}, "
                         f"Vertices: {detail_info['vertices']}, Solidity: {detail_info['solidity']:.2f}, AspectRatio: {detail_info['aspect_ratio']:.2f}, "
                         f"RectRatio: {detail_info['rect_ratio_str']}, BBox: {detail_info['bounding_rect']}")

        final_selected_contour = None
        if not candidate_outlines:
            logging.warning("--- No candidate contours found after all filtering stages. ---")
        else:
            candidate_outlines.sort(key=lambda c: (c['children'], c['area'], c['solidity']), reverse=True)
            
            logging.info("\n--- Top Candidate Outlines (Sorted by children, then area, then solidity): ---")
            for rank, cand in enumerate(candidate_outlines[:10]): 
                logging.info(f"  Rank {rank+1}: ID={cand['id']}, Parent={cand['parent_id']}, Area={cand['area']:.0f}, Children={cand['children']}, Solidity={cand['solidity']:.2f}, Vertices={cand['vertices']}, AspectRatio={cand['aspect_ratio']:.2f}, RectRatio={cand['rect_ratio_str']}, Rect=({cand['bounding_rect'][0]},{cand['bounding_rect'][1]},{cand['bounding_rect'][2]},{cand['bounding_rect'][3]})")
            
            max_sensible_area = image_area * max_building_candidate_area_ratio
            logging.info(f"--- Applying final selection logic with max_sensible_area_for_building_candidate: {max_sensible_area:.0f} (Ratio: {max_building_candidate_area_ratio}) ---")

            for cand in candidate_outlines:
                if cand['area'] <= max_sensible_area:
                    final_selected_contour = cand['contour_approx'] 
                    logging.info(f"--- Final Selected Outline (passed max_sensible_area check): ID={cand['id']}, Parent={cand['parent_id']}, Area={cand['area']:.0f}, Children={cand['children']} ---")
                    break 
                else:
                    logging.info(f"  Candidate ID={cand['id']} [Area:{cand['area']:.0f}] skipped: Exceeds max_sensible_area {max_sensible_area:.0f}.")
            
            if final_selected_contour is None:
                logging.warning("--- No candidate met the max_sensible_area criteria. No final outline selected. ---")
        
        return min_area_passed_details, final_selected_contour

    def find_building_outline(self, processed_image, all_contours, hierarchy, params):
        """
        Find the main building outline from a list of contours and hierarchy.
        Draws detailed info for all min_area_passed contours and highlights the final selected outline.
        """
        
        min_area_passed_details_list, final_selected_one = self.filter_and_select_outline(
            processed_image.shape, all_contours, hierarchy, params
        )

        if len(processed_image.shape) == 2 or processed_image.shape[2] == 1:
            outline_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        else:
            outline_image = processed_image.copy() 

        if min_area_passed_details_list: 
            logging.info(f"Drawing details for {len(min_area_passed_details_list)} contours that passed min_area_ratio filter.")
            
            for idx, details in enumerate(min_area_passed_details_list):
                contour_to_draw = details['contour_original'] 
                approx_contour = details['contour_approx']   
                contour_id = details['id']
                x_c, y_c, w_c, h_c = details['bounding_rect']
                
                color = (0, 255, 0) 
                bbox_color = (0, 255, 255) 
                vertex_color = (255, 0, 0) 
                text_color = (255, 255, 255) 

                cv2.drawContours(outline_image, [contour_to_draw], -1, color, 1) 
                cv2.rectangle(outline_image, (x_c, y_c), (x_c + w_c, y_c + h_c), bbox_color, 1)
                cv2.putText(outline_image, f"ID:{contour_id}", (x_c + 5, y_c + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

                if approx_contour is not None and len(approx_contour) > 0:
                    for point in approx_contour:
                        center = tuple(point[0])
                        cv2.circle(outline_image, center, 3, vertex_color, -1) 

        if final_selected_one is not None and len(final_selected_one) > 0:
            logging.info(f"Highlighting final selected outline.")
            highlight_color = (255, 0, 255) 
            highlight_thickness = 4      
            cv2.drawContours(outline_image, [final_selected_one], -1, highlight_color, highlight_thickness) 
        else:
            logging.warning("No final building outline was selected to highlight.")
            
        return outline_image, final_selected_one 

    def analyze_contours(self, contours):
        """
        Analyze contours to extract properties like area, perimeter, etc.
        """
        result = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
            aspect_ratio = float(w) / h if h > 0 else 0
            result.append({
                'id': i, 'area': area, 'perimeter': perimeter, 'points': len(contour),
                'bounds': (x, y, w, h), 'center': (cx, cy), 'aspect_ratio': aspect_ratio
            })
        return result

    def merge_overlapping_contours(self, contours, threshold=0.1):
        """
        Merge contours that have significant overlap.
        """
        if not contours or len(contours) < 2:
            return contours

        merged_contours = []
        merged_flags = [False] * len(contours)

        for i in range(len(contours)):
            if merged_flags[i]:
                continue

            current_contour = contours[i]
            max_x, max_y = 0, 0
            all_pts_for_mask = []
            for c_idx in range(len(contours)):
                if not merged_flags[c_idx]: # 아직 병합되지 않은 윤곽선들의 점만 고려
                    if contours[c_idx] is not None and len(contours[c_idx]) > 0:
                        for pt_array in contours[c_idx]:
                           all_pts_for_mask.append(pt_array[0])
            
            if not all_pts_for_mask: continue 

            # 윤곽선 점들로부터 최대 x, y 좌표 계산하여 마스크 크기 결정
            if all_pts_for_mask:
                max_x = max(p[0] for p in all_pts_for_mask) + 10
                max_y = max(p[1] for p in all_pts_for_mask) + 10
            else: # 모든 점이 비정상적인 경우 (이론상 발생하기 어려움)
                continue


            current_mask = np.zeros((max_y, max_x), dtype=np.uint8)
            cv2.drawContours(current_mask, [current_contour], -1, 255, -1)
            current_area = cv2.contourArea(current_contour)
            if current_area == 0: # 면적이 0인 윤곽선은 병합 시도에서 제외
                merged_flags[i] = True
                merged_contours.append(current_contour) # 그대로 추가하거나, 아예 제외할 수도 있음
                continue

            for j in range(i + 1, len(contours)):
                if merged_flags[j]:
                    continue

                other_contour = contours[j]
                if other_contour is None or len(other_contour) == 0:
                    merged_flags[j] = True # 유효하지 않은 윤곽선은 처리된 것으로 간주
                    continue

                other_mask = np.zeros((max_y, max_x), dtype=np.uint8)
                cv2.drawContours(other_mask, [other_contour], -1, 255, -1)
                other_area = cv2.contourArea(other_contour)
                if other_area == 0: # 면적이 0인 윤곽선은 병합 대상에서 제외
                    merged_flags[j] = True
                    continue


                intersection = cv2.bitwise_and(current_mask, other_mask)
                intersection_area = cv2.countNonZero(intersection)
                
                smaller_area = min(current_area, other_area)
                overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0

                if overlap_ratio > threshold:
                    combined_mask = cv2.bitwise_or(current_mask, other_mask)
                    
                    temp_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if temp_contours:
                        new_merged_contour = max(temp_contours, key=cv2.contourArea)
                        current_contour = new_merged_contour 
                        current_mask = np.zeros((max_y, max_x), dtype=np.uint8)
                        cv2.drawContours(current_mask, [current_contour], -1, 255, -1)
                        current_area = cv2.contourArea(current_contour)
                        if current_area == 0: # 병합 후 면적이 0이 되면 더 이상 진행하지 않음
                            break 


                    merged_flags[j] = True
            
            merged_contours.append(current_contour)
            merged_flags[i] = True 

        return merged_contours


    def simplify_contour(self, contour, epsilon_factor=0.01):
        """
        Simplify a contour using the Douglas-Peucker algorithm.
        """
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    def extract_walls(self, image, min_line_length=50, max_line_gap=5):
        """
        Extract wall lines from a floor plan.
        """
        if len(image.shape) == 3: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
        elif np.max(image) > 1 and len(np.unique(image)) > 2 : 
             _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        else: 
            binary = image.copy()
            if np.max(binary) == 1: binary = binary * 255 
        
        if binary.dtype != np.uint8: binary = binary.astype(np.uint8)

        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 180, threshold=50,
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        if len(image.shape) == 2 or image.shape[2] == 1:
             wall_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) 
        else:
             wall_image = image.copy()


        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(wall_image, (x1, y1), (x2, y2), (0, 255, 255), 2) 
        return wall_image, lines if lines is not None else []

    def extract_rooms(self, image, min_area=1000, max_area=100000):
        """
        Extract room contours from a floor plan.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        processed_binary = image.copy()
        if np.max(processed_binary) == 1: processed_binary = processed_binary * 255
        if processed_binary.dtype != np.uint8: processed_binary = processed_binary.astype(np.uint8)

        inverted_binary = cv2.bitwise_not(processed_binary) 

        contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        room_contours = []
        if hierarchy is not None:
            hierarchy = hierarchy[0] # OpenCV returns hierarchy inside a list
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x,y,w,h = cv2.boundingRect(contour)
                    img_h, img_w = image.shape[:2]
                    # 경계에서 너무 가까운 영역 제외 (약간의 여유를 둠)
                    if x > 5 and y > 5 and (x+w) < img_w-5 and (y+h) < img_h-5 :
                        aspect_ratio = w/float(h) if h > 0 else 0
                        if 0.2 < aspect_ratio < 5.0: 
                            room_contours.append(contour)

        if len(image.shape) == 2 or image.shape[2] == 1:
            room_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            room_image = image.copy()

        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
        for i, contour in enumerate(room_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(room_image, [contour], -1, color, 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(room_image, f"R{i+1}", (cx - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return room_image, room_contours
    
    def extract_rooms_from_roi(self, image, building_contour, min_area=10000, max_area=100000,
                               wall_closing_kernel_size=1, # Kernel size for closing operation on walls
                               space_opening_kernel_size=1, # Kernel size for opening operation on space (formerly morph_kernel_size)
                               dist_transform_threshold_ratio=0.1, roi_margin=0,
                               room_simplify_epsilon_factor=0.02, 
                               min_area_for_missed_rooms_ratio=0.5):
        
        debug_image_dir = "debug_room_extraction"
        if not os.path.exists(debug_image_dir):
            os.makedirs(debug_image_dir)
            logging.info(f"Created directory: {debug_image_dir}")

        logging.info(f"--- Starting Room Extraction from ROI using Flood Fill ---")
        if building_contour is None or len(building_contour) == 0:
            logging.warning("Input building_contour for ROI is None or empty. Skipping room extraction.")
            return image.copy(), []

        x_orig_bbox, y_orig_bbox, w_orig_bbox, h_orig_bbox = cv2.boundingRect(building_contour)
        logging.debug(f"Original ROI BoundingBox: x={x_orig_bbox}, y={y_orig_bbox}, w={w_orig_bbox}, h={h_orig_bbox}")

        img_h_main, img_w_main = image.shape[:2]

        x_roi = max(0, x_orig_bbox - roi_margin)
        y_roi = max(0, y_orig_bbox - roi_margin)
        
        w_roi = min(img_w_main - x_roi, w_orig_bbox + 2 * roi_margin)
        h_roi = min(img_h_main - y_roi, h_orig_bbox + 2 * roi_margin)

        if w_roi <= 0 or h_roi <= 0:
            logging.error(f"Invalid ROI size after margin: w={w_roi}, h={h_roi}. Skipping.")
            return image.copy(), []
            
        logging.debug(f"ROI with Margin: x_roi={x_roi}, y_roi={y_roi}, w_roi={w_roi}, h_roi={h_roi}")

        roi_original_with_margin = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        logging.debug(f"ROI extracted with margin. ROI shape: {roi_original_with_margin.shape}")
        cv2.imwrite(os.path.join(debug_image_dir, "01_roi_original_with_margin.png"), roi_original_with_margin)

        binary_roi_internal = None
        # Initial binarization to get "space=255, wall=0"
        if len(roi_original_with_margin.shape) == 3:
            gray_roi = cv2.cvtColor(roi_original_with_margin, cv2.COLOR_BGR2GRAY)
            _, binary_roi_internal = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if np.mean(binary_roi_internal) < 128: # If Otsu made space black
                binary_roi_internal = cv2.bitwise_not(binary_roi_internal)
            logging.debug("ROI converted from color to binary (attempted space=255, wall=0).")
        elif np.max(roi_original_with_margin) == 1 and len(np.unique(roi_original_with_margin)) == 2 :
            binary_roi_internal = (roi_original_with_margin * 255).astype(np.uint8)
            if np.mean(binary_roi_internal) < 128: 
                 binary_roi_internal = cv2.bitwise_not(binary_roi_internal)
            logging.debug("ROI converted from 0/1 binary to 0/255 binary (attempted space=255, wall=0).")
        elif len(np.unique(roi_original_with_margin)) > 2 : # Grayscale
             _, binary_roi_internal = cv2.threshold(roi_original_with_margin, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
             if np.mean(binary_roi_internal) < 128:
                binary_roi_internal = cv2.bitwise_not(binary_roi_internal)
             logging.debug("ROI converted from grayscale to binary (attempted space=255, wall=0).")
        else: # Already 0/255 binary
            binary_roi_internal = roi_original_with_margin.copy()
            unique_vals, counts = np.unique(binary_roi_internal, return_counts=True)
            val_map = dict(zip(unique_vals, counts))
            if val_map.get(0,0) > val_map.get(255,0): # If space is black
                binary_roi_internal = cv2.bitwise_not(binary_roi_internal)
                logging.debug("ROI was inverted to make space=255, wall=0.")
            else:
                logging.debug("ROI is already binary, assumed space=255, wall=0.")
        
        logging.debug(f"After initial binarization, unique values: {np.unique(binary_roi_internal, return_counts=True)}")
        cv2.imwrite(os.path.join(debug_image_dir, "02_binary_roi_initial.png"), binary_roi_internal)

        # Step 1: Improve wall continuity (on inverted image)
        if wall_closing_kernel_size > 0 and binary_roi_internal is not None:
            walls_mask = cv2.bitwise_not(binary_roi_internal) # Now walls=255, space=0
            kernel_wall_closing = np.ones((wall_closing_kernel_size, wall_closing_kernel_size), np.uint8)
            closed_walls = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, kernel_wall_closing)
            binary_roi_internal = cv2.bitwise_not(closed_walls) # Back to space=255, walls=0
            logging.debug(f"Applied MORPH_CLOSE to walls (inverted) with kernel {wall_closing_kernel_size}.")
            logging.debug(f"After wall closing, unique values: {np.unique(binary_roi_internal, return_counts=True)}")
            cv2.imwrite(os.path.join(debug_image_dir, "03_binary_roi_after_wall_closing.png"), binary_roi_internal)
        elif binary_roi_internal is None:
            logging.error("binary_roi_internal is None before wall closing. Skipping.")
            return image.copy(), []
        else:
            logging.debug("Wall closing was skipped.")
            cv2.imwrite(os.path.join(debug_image_dir, "03_binary_roi_wall_closing_skipped.png"), binary_roi_internal)


        # Step 2: Clean up space area (remove noise, smooth boundaries)
        if space_opening_kernel_size > 0 and binary_roi_internal is not None:
            kernel_space_opening = np.ones((space_opening_kernel_size, space_opening_kernel_size), np.uint8)
            binary_roi_internal = cv2.morphologyEx(binary_roi_internal, cv2.MORPH_OPEN, kernel_space_opening)
            logging.debug(f"Applied MORPH_OPEN to space with kernel size {space_opening_kernel_size}.")
            logging.debug(f"After space opening, unique values: {np.unique(binary_roi_internal, return_counts=True)}")
            cv2.imwrite(os.path.join(debug_image_dir, "04_binary_roi_after_space_opening.png"), binary_roi_internal)
        elif binary_roi_internal is None: # Should not happen if previous check passed
            logging.error("binary_roi_internal is None before space opening. Skipping.")
            return image.copy(), []
        else:
            logging.debug("Space opening was skipped.")
            cv2.imwrite(os.path.join(debug_image_dir, "04_binary_roi_space_opening_skipped.png"), binary_roi_internal)


        if cv2.countNonZero(binary_roi_internal) == 0:
            logging.error("binary_roi_internal is all black after morphological operations. Cannot proceed.")
            return image.copy(), []
            
        dist_transform = cv2.distanceTransform(binary_roi_internal, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(debug_image_dir, "05_dist_transform_normalized.png"), (dist_transform * 255).astype(np.uint8))
        
        _, max_dist_val, _, _ = cv2.minMaxLoc(dist_transform)
        threshold_dist = max_dist_val * dist_transform_threshold_ratio
        
        _, seed_mask = cv2.threshold(dist_transform, threshold_dist, 255, cv2.THRESH_BINARY)
        seed_mask = seed_mask.astype(np.uint8)
        logging.debug(f"Distance transform applied. Max dist: {max_dist_val:.4f}, Threshold for seeds: {threshold_dist:.4f}. Num seed candidate pixels: {cv2.countNonZero(seed_mask)}")
        cv2.imwrite(os.path.join(debug_image_dir, "06_seed_mask.png"), seed_mask)

        seed_points_candidates, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        seed_points = []
        for seed_area_contour in seed_points_candidates:
            M = cv2.moments(seed_area_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if 0 <= cy < binary_roi_internal.shape[0] and 0 <= cx < binary_roi_internal.shape[1]:
                    if binary_roi_internal[cy, cx] == 255:
                        seed_points.append((cx, cy))
                else:
                    logging.warning(f"Seed candidate center ({cx},{cy}) is out of bounds for binary_roi_internal shape {binary_roi_internal.shape}")
        
        if not seed_points and cv2.countNonZero(seed_mask) > 0 :
             coords = np.column_stack(np.where(seed_mask > 0))
             if len(coords) > 0:
                 sample_indices = np.linspace(0, len(coords) - 1, min(20, len(coords)), dtype=int)
                 for idx in sample_indices:
                     py, px = coords[idx] 
                     if 0 <= py < binary_roi_internal.shape[0] and 0 <= px < binary_roi_internal.shape[1]:
                        if binary_roi_internal[py, px] == 255: 
                            seed_points.append((px, py)) 
                 seed_points = sorted(list(set(seed_points)))


        logging.info(f"Found {len(seed_points)} potential seed points for flood fill.")

        floodfill_mask_template = np.zeros((h_roi + 2, w_roi + 2), np.uint8)
        room_id_counter = 1 
        
        processed_overall_mask = np.zeros_like(binary_roi_internal, dtype=np.uint8) 
        room_contours_from_floodfill = []

        for seed_idx, (seed_x, seed_y) in enumerate(seed_points):
            if processed_overall_mask[seed_y, seed_x] != 0:
                logging.debug(f"Seed point ({seed_x},{seed_y}) is already part of a filled region. Skipping.")
                continue

            logging.debug(f"Attempting flood fill from seed ({seed_x},{seed_y}) with potential room_id {room_id_counter}")
            
            current_iter_floodfill_mask = floodfill_mask_template.copy() 

            num_filled_pixels, image_after_fill, filled_mask_output, _ = cv2.floodFill(
                binary_roi_internal.copy(), 
                current_iter_floodfill_mask, 
                (seed_x, seed_y),
                newVal=0, 
                loDiff=0, upDiff=0,
                flags= (255 << 8) | cv2.FLOODFILL_MASK_ONLY 
            )
            
            single_room_mask_roi_size = filled_mask_output[1:-1, 1:-1].copy()

            if num_filled_pixels > 0:
                logging.debug(f"  Flood fill for seed_idx {seed_idx} (room_id {room_id_counter}) filled {num_filled_pixels} pixels.")
                cv2.imwrite(os.path.join(debug_image_dir, f"07_filled_mask_room_seed{seed_idx}.png"), single_room_mask_roi_size) # Debug image index updated
                
                contours_in_filled_area, _ = cv2.findContours(single_room_mask_roi_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours_in_filled_area:
                    room_contour_roi_raw = max(contours_in_filled_area, key=cv2.contourArea)
                    area = cv2.contourArea(room_contour_roi_raw)
                    logging.debug(f"    Found raw contour for room_id {room_id_counter} (FloodFill). Area: {area:.2f}, Vertices: {len(room_contour_roi_raw)}")

                    simplified_contour_roi = room_contour_roi_raw
                    if room_simplify_epsilon_factor > 0 and len(room_contour_roi_raw) > 0:
                        perimeter = cv2.arcLength(room_contour_roi_raw, True)
                        epsilon = room_simplify_epsilon_factor * perimeter
                        simplified_contour_roi = cv2.approxPolyDP(room_contour_roi_raw, epsilon, True)
                        logging.debug(f"    Simplified contour for room_id {room_id_counter}. New Vertices: {len(simplified_contour_roi)}, Epsilon: {epsilon:.2f}")
                    else:
                        logging.debug(f"    No simplification applied for room_id {room_id_counter}.")
                                   
                    contour_orig_coords = simplified_contour_roi + np.array([x_roi, y_roi]).reshape(-1, 1, 2)
                    room_contours_from_floodfill.append({'id': room_id_counter, 'contour': contour_orig_coords, 'area': area, 'type': 'floodfill'})
                    logging.info(f"    Accepted room_id {room_id_counter} (FloodFill, SIMPLIFIED). Area: {area:.2f}")
                    
                    processed_overall_mask = cv2.bitwise_or(processed_overall_mask, single_room_mask_roi_size)
                    room_id_counter += 1
                else:
                    logging.debug(f"  No contour found for flood-filled area of room_id {room_id_counter}.")
            else:
                logging.debug(f"  Flood fill for seed_idx {seed_idx} (room_id {room_id_counter}) did not fill any pixels.")

        logging.info("--- Attempting to find missed rooms in unfilled areas ---")
        unfilled_area_mask = cv2.bitwise_not(processed_overall_mask)
        unfilled_area_mask = cv2.bitwise_and(unfilled_area_mask, binary_roi_internal) # Use the final state of binary_roi_internal
        cv2.imwrite(os.path.join(debug_image_dir, "08_unfilled_area_mask_for_missed.png"), unfilled_area_mask) # Debug image index updated

        # ── Step 8.1: 틈 메우기 (Close) ──
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        unfilled_area_mask = cv2.morphologyEx(unfilled_area_mask, cv2.MORPH_CLOSE, kernel_close)
        cv2.imwrite(os.path.join(debug_image_dir, "08b_unfilled_closed.png"), unfilled_area_mask)

        # ── Step 8.2: 살짝 팽창 (Dilate) ──
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        unfilled_area_mask = cv2.dilate(unfilled_area_mask, kernel_dilate, iterations=1)
        cv2.imwrite(os.path.join(debug_image_dir, "08c_unfilled_dilated.png"), unfilled_area_mask)

        missed_room_contours_roi, _ = cv2.findContours(unfilled_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.info(f"Found {len(missed_room_contours_roi)} potential missed room contours.")

        min_effective_area_for_missed = min_area * min_area_for_missed_rooms_ratio

        for missed_idx, contour_roi_raw in enumerate(missed_room_contours_roi):
            area = cv2.contourArea(contour_roi_raw)
            logging.debug(f"  Missed Candidate {missed_idx}: Area={area:.2f}, Vertices: {len(contour_roi_raw)}")

            if area >= min_effective_area_for_missed and area <=max_area :
                x_r_bbox, y_r_bbox, w_r_bbox, h_r_bbox = cv2.boundingRect(contour_roi_raw)
                aspect_ratio = w_r_bbox / float(h_r_bbox) if h_r_bbox > 0 else 0
                if 0.05 < aspect_ratio < 20.0: 
                    
                    simplified_contour_roi = contour_roi_raw
                    if room_simplify_epsilon_factor > 0 and len(contour_roi_raw) > 0:
                        perimeter = cv2.arcLength(contour_roi_raw, True)
                        epsilon = room_simplify_epsilon_factor * perimeter 
                        simplified_contour_roi = cv2.approxPolyDP(contour_roi_raw, epsilon, True)
                        logging.debug(f"    Simplified contour for missed room {missed_idx}. New Vertices: {len(simplified_contour_roi)}")
                    else:
                        logging.debug(f"    No simplification for missed room {missed_idx}.")

                    contour_orig_coords = simplified_contour_roi + np.array([x_roi, y_roi]).reshape(-1, 1, 2)
                    room_contours_from_floodfill.append({'id': f"M{room_id_counter}", 'contour': contour_orig_coords, 'area': area, 'type': 'missed'})
                    logging.info(f"    Accepted Missed Room M{room_id_counter}. Area: {area:.2f}, Aspect Ratio: {aspect_ratio:.2f}")
                    room_id_counter +=1 
                else:
                    logging.debug(f"  Missed Candidate {missed_idx} filtered by aspect ratio: {aspect_ratio:.2f}")
            else:
                logging.debug(f"  Missed Candidate {missed_idx} filtered by area: {area:.2f} (Min_eff: {min_effective_area_for_missed}, Max: {max_area})")

        room_image_display = image.copy()
        if len(room_image_display.shape) == 2 or room_image_display.shape[2] == 1:
            room_image_display = cv2.cvtColor(room_image_display, cv2.COLOR_GRAY2BGR)

        final_room_contours = []
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255),
                  (128,0,0), (0,128,0), (0,0,128), (128,128,0), (0,128,128), (128,0,128),
                  (200,100,50), (50,200,100), (100,50,200), (200,50,100)] 

        logging.info(f"Drawing {len(room_contours_from_floodfill)} detected rooms (FloodFill + Missed, SIMPLIFIED).")
        for i, room_data in enumerate(room_contours_from_floodfill):
            contour_item_draw = room_data['contour']
            room_display_id = room_data['id'] 
            
            color_idx = i
            if isinstance(room_display_id, str) and room_display_id.startswith("M"):
                 color_idx = i + len(colors)//2 
            color = colors[color_idx % len(colors)]

            cv2.drawContours(room_image_display, [contour_item_draw], -1, color, 2)
            M = cv2.moments(contour_item_draw)
            if M["m00"] != 0:
                cx_draw = int(M["m10"] / M["m00"])
                cy_draw = int(M["m01"] / M["m00"])
                cv2.putText(room_image_display, f"R{room_display_id}", (cx_draw - 15, cy_draw + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            final_room_contours.append(contour_item_draw)
        
        cv2.imwrite(os.path.join(debug_image_dir, "09_final_room_detection_overlay_all_steps.png"), room_image_display) # New filename

        logging.info(f"--- Finished Room Extraction. Total rooms found: {len(final_room_contours)} (FloodFill + Missed, SIMPLIFIED). ---")
        return room_image_display, final_room_contours
    
    def save_contours_to_dxf(self, contours, filename):
        """
        Save contours to a DXF file format (for CAD software).
        """
        if not contours: 
            print("No contours to save.")
            return False
            
        try:
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()

            # contours가 단일 윤곽선(np.ndarray)이거나 윤곽선 리스트(list of np.ndarray)일 수 있음
            # final_selected_contour는 단일 윤곽선일 가능성이 높음
            if isinstance(contours, np.ndarray) and contours.ndim >=2 : 
                # 단일 윤곽선이지만, 리스트로 감싸서 처리하면 아래 로직과 통일 가능
                contours_list = [contours]
            elif isinstance(contours, list):
                contours_list = contours
            else:
                print(f"Unsupported contours type for DXF saving: {type(contours)}")
                return False

            for i, contour in enumerate(contours_list):
                if contour is not None and len(contour) > 0:
                    # contour의 각 점은 [[x, y]] 형태일 수 있으므로 (x, y)로 변환
                    points = [tuple(point[0]) if isinstance(point, np.ndarray) and point.shape == (1,2) else tuple(point) for point in contour]
                    msp.add_lwpolyline(points, close=True, dxfattribs={'layer': f'contour_{i}'})
            
            doc.saveas(filename)
            return True
        except ImportError:
            print("ezdxf package not installed. Please install it: pip install ezdxf")
            return False
        except Exception as e:
            print(f"Error saving to DXF: {e}")
            import traceback
            traceback.print_exc()
            return False