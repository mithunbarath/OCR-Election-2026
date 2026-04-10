import cv2
import numpy as np

def extract_cards_from_page(page_img):
    """
    Slices the high-res page image into 30 individual voter cards.
    Assumes standard ECI layout (3 columns, 10 rows).
    """
    img_h, img_w = page_img.shape[:2]
    
    # Standard offsets for header/footer (approximate percentages)
    header_offset = int(img_h * 0.11)
    footer_offset = int(img_h * 0.04)
    grid_h = img_h - header_offset - footer_offset
    
    row_height = grid_h // 10
    col_width = img_w // 3
    
    cards = []
    
    for row in range(10):
        for col in range(3):
            y_start = header_offset + (row * row_height)
            y_end = y_start + row_height
            x_start = col * col_width
            x_end = x_start + col_width
            
            card_crop = page_img[y_start:y_end, x_start:x_end]
            cards.append(card_crop)
            
    return cards

def get_sub_regions(card_img):
    """
    Breaks a single voter card into predefined Regions of Interest (ROI).
    Returns a dictionary of cropped images.
    """
    h, w = card_img.shape[:2]
    
    rois = {
        # Top-left portion
        "serial_no": card_img[0 : int(h * 0.15), 0 : int(w * 0.35)],
        
        # Top-right portion
        "epic_id": card_img[0 : int(h * 0.15), int(w * 0.35) : w],
        
        # Middle-right portion (Name and Relation details usually avoid the left-side photo)
        "name_rel": card_img[int(h * 0.15) : int(h * 0.65), int(w * 0.25) : w],
        
        # Bottom-right portion (House No, Age, Gender)
        # Note: Age and Gender are sometimes scattered near the bottom edge
        "bottom_details": card_img[int(h * 0.65) : int(h * 0.95), int(w * 0.25) : w]
    }
    
    return rois
