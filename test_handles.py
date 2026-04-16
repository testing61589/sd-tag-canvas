#!/usr/bin/env python3
"""
Test script to understand crop handle positioning issues
"""

from PySide6.QtCore import QRect, QPoint
from PySide6.QtGui import QPainter, QPixmap

# Simulate the crop handle logic
CROP_HANDLE_SIZE = 12
MIN_CROP_SIZE = 10

def get_handle_rects(screen_rect):
    """Get rectangles for all crop handles"""
    handles = {}

    # Corner handles
    handles['top-left'] = QRect(screen_rect.left() - CROP_HANDLE_SIZE//2,
                               screen_rect.top() - CROP_HANDLE_SIZE//2,
                               CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    handles['top-right'] = QRect(screen_rect.right() - CROP_HANDLE_SIZE//2,
                                screen_rect.top() - CROP_HANDLE_SIZE//2,
                                CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    handles['bottom-left'] = QRect(screen_rect.left() - CROP_HANDLE_SIZE//2,
                                  screen_rect.bottom() - CROP_HANDLE_SIZE//2,
                                  CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    handles['bottom-right'] = QRect(screen_rect.right() - CROP_HANDLE_SIZE//2,
                                   screen_rect.bottom() - CROP_HANDLE_SIZE//2,
                                   CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    # Edge handles
    handles['top'] = QRect(screen_rect.center().x() - CROP_HANDLE_SIZE//2,
                          screen_rect.top() - CROP_HANDLE_SIZE//2,
                          CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    handles['bottom'] = QRect(screen_rect.center().x() - CROP_HANDLE_SIZE//2,
                             screen_rect.bottom() - CROP_HANDLE_SIZE//2,
                             CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    handles['left'] = QRect(screen_rect.left() - CROP_HANDLE_SIZE//2,
                           screen_rect.center().y() - CROP_HANDLE_SIZE//2,
                           CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    handles['right'] = QRect(screen_rect.right() - CROP_HANDLE_SIZE//2,
                            screen_rect.center().y() - CROP_HANDLE_SIZE//2,
                            CROP_HANDLE_SIZE, CROP_HANDLE_SIZE)

    return handles

def test_handle_positions():
    """Test handle positions for different crop sizes"""

    # Test case 1: Large crop
    print("Test 1: Large crop (100x100) at position (50, 50)")
    screen_rect_large = QRect(50, 50, 100, 100)
    handles_large = get_handle_rects(screen_rect_large)
    for name, rect in handles_large.items():
        print(f"  {name}: {rect}")

    # Check if handles overlap
    print("\n  Checking for overlaps:")
    for name1, rect1 in handles_large.items():
        for name2, rect2 in handles_large.items():
            if name1 != name2 and rect1.intersects(rect2):
                print(f"    {name1} overlaps with {name2}")

    # Test case 2: Small crop (at minimum size)
    print("\nTest 2: Small crop (10x10) at position (0, 0)")
    screen_rect_small = QRect(0, 0, 10, 10)
    handles_small = get_handle_rects(screen_rect_small)
    for name, rect in handles_small.items():
        print(f"  {name}: {rect}")

    # Check if handles are within bounds
    print("\n  Checking handle positions:")
    for name, rect in handles_small.items():
        # Handle extends beyond crop rectangle
        if name in ['top-left', 'top', 'left']:
            if rect.left() < 0 or rect.top() < 0:
                print(f"    {name}: extends outside (negative coordinates)")
        if name in ['top-right', 'top', 'right']:
            if rect.right() > 10:
                print(f"    {name}: extends outside (right > 10)")
        if name in ['bottom-left', 'bottom', 'left']:
            if rect.bottom() > 10:
                print(f"    {name}: extends outside (bottom > 10)")
        if name in ['bottom-right', 'bottom', 'right']:
            if rect.right() > 10 or rect.bottom() > 10:
                print(f"    {name}: extends outside")

    # Test case 3: Very small crop (below minimum size)
    print("\nTest 3: Very small crop (5x5) at position (0, 0)")
    screen_rect_tiny = QRect(0, 0, 5, 5)
    handles_tiny = get_handle_rects(screen_rect_tiny)
    print("  Handle positions:")
    for name, rect in handles_tiny.items():
        print(f"    {name}: {rect}")

    # Check if handles can overlap each other
    print("\n  Checking for handle overlaps:")
    overlaps = []
    handle_names = list(handles_tiny.keys())
    for i, name1 in enumerate(handle_names):
        for name2 in handle_names[i+1:]:
            rect1 = handles_tiny[name1]
            rect2 = handles_tiny[name2]
            if rect1.intersects(rect2):
                overlaps.append((name1, name2))

    if overlaps:
        print(f"    Found {len(overlaps)} overlaps:")
        for name1, name2 in overlaps:
            print(f"      {name1} <-> {name2}")
    else:
        print("    No overlaps found")

if __name__ == "__main__":
    test_handle_positions()