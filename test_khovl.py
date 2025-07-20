#!/usr/bin/env python3
"""
Test script for TikTok Skip Bot functionality
This script tests the core functions without requiring TikTok to be open
"""

import sys
import os
import logging
import numpy as np
import cv2

# Add the current directory to Python path to import khovl
sys.path.insert(0, '/home/runner/work/CBB/CBB')

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_test_image(width=800, height=600, text="TEST"):
    """Create a test image with text"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(128)  # Gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # White text
    thickness = 3
    
    # Calculate text size and position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img

def create_test_templates():
    """Create test template images"""
    print("🔧 Creating test templates...")
    
    # Create test live templates
    live_template = create_test_image(100, 50, "LIVE")
    cv2.imwrite("/home/runner/work/CBB/CBB/templates/live/test_live_badge.png", live_template)
    
    # Create test ad templates  
    ad_template = create_test_image(120, 60, "AD")
    cv2.imwrite("/home/runner/work/CBB/CBB/templates/ads/test_ad_banner.png", ad_template)
    
    print("✅ Test templates created")

def test_template_loading():
    """Test template loading functionality"""
    print("\n🧪 Testing template loading...")
    
    try:
        from khovl import TikTokSkipBot
        bot = TikTokSkipBot()
        
        print(f"✅ Ad templates loaded: {len(bot.ad_templates)}")
        print(f"✅ Live templates loaded: {len(bot.live_templates)}")
        print(f"✅ Live keywords: {len(bot.live_keywords)}")
        
        return bot
    except Exception as e:
        print(f"❌ Template loading failed: {e}")
        return None

def test_image_detection(bot):
    """Test image detection functionality"""
    print("\n🧪 Testing image detection...")
    
    try:
        # Create test screenshot with LIVE text
        test_screenshot = create_test_image(800, 600, "LIVE TEST")
        
        # Test live detection with lowered threshold
        detected, match_val, template_name = bot.detect_image(
            test_screenshot, bot.live_templates, "live", threshold=0.4
        )
        
        print(f"🔍 Live detection result: {detected}")
        print(f"📊 Best match value: {match_val:.4f}")
        print(f"📁 Best template: {template_name}")
        
        # Test ad detection
        ad_detected, ad_match_val, ad_template = bot.detect_image(
            test_screenshot, bot.ad_templates, "ad", threshold=0.6
        )
        
        print(f"🔍 Ad detection result: {ad_detected}")
        print(f"📊 Ad match value: {ad_match_val:.4f}")
        
    except Exception as e:
        print(f"❌ Image detection test failed: {e}")

def test_text_detection(bot):
    """Test OCR text detection functionality"""
    print("\n🧪 Testing OCR text detection...")
    
    try:
        # Create test screenshot with live text
        test_screenshot = create_test_image(800, 600, "LIVE STREAM")
        
        # Test OCR detection
        detected, detected_text = bot.detect_text(test_screenshot)
        
        print(f"🔍 OCR detection result: {detected}")
        print(f"📝 Detected text: {detected_text}")
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        print("Note: OCR testing may fail without proper tesseract installation")

def test_window_detection(bot):
    """Test window detection functionality"""
    print("\n🧪 Testing window detection...")
    
    try:
        windows = bot.get_tiktok_windows()
        print(f"🔍 TikTok windows found: {len(windows)}")
        
        for i, window in enumerate(windows):
            print(f"  Window {i+1}: {window.title}")
            
    except Exception as e:
        print(f"❌ Window detection test failed: {e}")
        print("Note: Window detection may fail in headless environments")

def test_logging_levels():
    """Test that logging is properly configured"""
    print("\n🧪 Testing logging configuration...")
    
    # Test different log levels
    logging.debug("🔍 This DEBUG message should NOT appear (level=INFO)")
    logging.info("✅ This INFO message SHOULD appear")
    logging.warning("⚠️ This WARNING message should appear")
    logging.error("❌ This ERROR message should appear")
    
    print("✅ Logging test complete - check above for proper log levels")

def main():
    """Run all tests"""
    print("🚀 Starting TikTok Skip Bot Tests...")
    print("=" * 60)
    
    # Create test templates
    create_test_templates()
    
    # Test logging
    test_logging_levels()
    
    # Test template loading
    bot = test_template_loading()
    
    if bot:
        # Test image detection
        test_image_detection(bot)
        
        # Test text detection (may fail without tesseract)
        test_text_detection(bot)
        
        # Test window detection (may fail in headless)
        test_window_detection(bot)
    
    print("\n" + "=" * 60)
    print("🎉 Test suite completed!")
    print("💡 Check the logs above for detailed results")
    print("📝 Full logs are also saved to 'tiktok_skip.log'")

if __name__ == "__main__":
    main()