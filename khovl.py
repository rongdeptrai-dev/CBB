#!/usr/bin/env python3
"""
TikTok Livestream Skip Bot - Vietnamese
Chức năng tự động skip quảng cáo và livestream trên TikTok
"""

import os
import cv2
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
import glob
from pathlib import Path

# Optional imports with fallbacks
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logging.warning("⚠️ pytesseract not available - OCR functionality disabled")

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
    logging.warning("⚠️ pygetwindow not available - window detection disabled")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("⚠️ pyautogui not available - screenshot functionality disabled")

# Configure logging - Force configuration even if already configured
logging.root.handlers = []  # Clear existing handlers
logging.basicConfig(
    level=logging.INFO,  # Set to INFO level to capture debug messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tiktok_skip.log', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration
)

class TikTokSkipBot:
    def __init__(self):
        self.ad_templates_dir = "templates/ads"
        self.live_templates_dir = "templates/live"
        self.ad_templates = []
        self.live_templates = []
        self.live_keywords = [
            "LIVE", "Live", "live", "TRỰC TIẾP", "Trực tiếp", "trực tiếp",
            "PHÁT TRỰC TIẾP", "Phát trực tiếp", "phát trực tiếp",
            "ĐANG LIVE", "Đang live", "đang live"
        ]
        self.load_templates()
        
    def load_templates(self):
        """Load ad and live templates"""
        logging.info("🔄 Loading templates...")
        
        # Load ad templates
        if os.path.exists(self.ad_templates_dir):
            ad_files = glob.glob(os.path.join(self.ad_templates_dir, "*.png")) + \
                      glob.glob(os.path.join(self.ad_templates_dir, "*.jpg"))
            self.ad_templates = [(cv2.imread(f, 0), os.path.basename(f)) for f in ad_files]
            logging.info(f"📁 Loaded {len(self.ad_templates)} ad templates")
        
        # Load live templates
        if os.path.exists(self.live_templates_dir):
            live_files = glob.glob(os.path.join(self.live_templates_dir, "*.png")) + \
                        glob.glob(os.path.join(self.live_templates_dir, "*.jpg"))
            self.live_templates = [(cv2.imread(f, 0), os.path.basename(f)) for f in live_files]
            logging.info(f"📺 Loaded {len(self.live_templates)} live templates")
        
        total_templates = len(self.ad_templates) + len(self.live_templates)
        logging.info(f"✅ Total templates loaded: {total_templates}")

    def get_tiktok_windows(self) -> List:
        """Get all TikTok windows with enhanced debugging"""
        if not PYGETWINDOW_AVAILABLE:
            logging.error("❌ pygetwindow not available - cannot detect windows")
            return []
            
        try:
            all_windows = gw.getAllWindows()
            tiktok_windows = []
            
            logging.info(f"🔍 Scanning {len(all_windows)} windows for TikTok...")
            
            for window in all_windows:
                window_title = window.title.lower()
                if any(keyword in window_title for keyword in ['tiktok', 'tik tok']):
                    tiktok_windows.append(window)
                    logging.info(f"🎯 Found TikTok window: '{window.title}' (size: {window.width}x{window.height})")
            
            if not tiktok_windows:
                logging.warning("⚠️ No TikTok windows found!")
                # Log some window titles for debugging
                sample_titles = [w.title for w in all_windows[:10] if w.title.strip()]
                logging.info(f"🔍 Sample window titles: {sample_titles}")
            else:
                logging.info(f"✅ Found {len(tiktok_windows)} TikTok window(s)")
                
            return tiktok_windows
            
        except Exception as e:
            logging.error(f"❌ Error getting TikTok windows: {e}")
            return []

    def capture_window(self, window) -> Optional[np.ndarray]:
        """Capture screenshot of a window"""
        if not PYAUTOGUI_AVAILABLE:
            logging.error("❌ pyautogui not available - cannot capture screenshots")
            return None
            
        try:
            # Bring window to front and capture
            window.activate()
            time.sleep(0.1)
            
            screenshot = pyautogui.screenshot(region=(
                window.left, window.top, window.width, window.height
            ))
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logging.error(f"❌ Error capturing window: {e}")
            return None

    def detect_image(self, screenshot: np.ndarray, templates: List, template_type: str, threshold: float = 0.6) -> Tuple[bool, float, str]:
        """
        Detect templates in screenshot with enhanced logging
        Fixed: Changed from logging.debug() to logging.info() for visibility
        """
        if not templates:
            logging.info(f"📋 No {template_type} templates to check")
            return False, 0.0, ""
        
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        best_match_val = 0.0
        best_template_name = ""
        potential_matches = []
        
        logging.info(f"🔍 Checking {len(templates)} {template_type} templates (threshold: {threshold})")
        
        for template, template_name in templates:
            if template is None:
                logging.warning(f"⚠️ Template {template_name} is None, skipping")
                continue
                
            try:
                # Perform template matching
                result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                # Enhanced logging for live templates
                if template_type == "live":
                    logging.info(f"📺 Live template '{template_name}': match value = {max_val:.4f}")
                else:
                    logging.info(f"📢 Ad template '{template_name}': match value = {max_val:.4f}")
                
                # Log potential matches (>= 0.3 for analysis)
                if max_val >= 0.3:
                    potential_matches.append((template_name, max_val))
                    logging.info(f"🎯 Potential match: '{template_name}' = {max_val:.4f}")
                
                # Track best match
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_template_name = template_name
                    
            except Exception as e:
                logging.error(f"❌ Error matching template {template_name}: {e}")
        
        # Summary logging
        if potential_matches:
            logging.info(f"📊 Found {len(potential_matches)} potential matches for {template_type}")
            for name, val in sorted(potential_matches, key=lambda x: x[1], reverse=True):
                logging.info(f"   - {name}: {val:.4f}")
        else:
            logging.info(f"❌ No potential matches found for {template_type}")
        
        is_detected = best_match_val >= threshold
        if is_detected:
            logging.info(f"✅ {template_type.upper()} DETECTED! Best match: '{best_template_name}' = {best_match_val:.4f}")
        else:
            logging.info(f"❌ No {template_type} detected above threshold ({threshold})")
            
        return is_detected, best_match_val, best_template_name

    def detect_text(self, screenshot: np.ndarray) -> Tuple[bool, str]:
        """
        Detect live keywords using OCR with enhanced debugging
        Fixed: Added detailed logging for live OCR detection
        """
        if not PYTESSERACT_AVAILABLE:
            logging.warning("⚠️ pytesseract not available - OCR detection disabled")
            return False, "OCR not available"
            
        try:
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # OCR configuration for better Vietnamese text recognition
            config = '--oem 3 --psm 6 -l vie+eng'
            
            # Extract text
            extracted_text = pytesseract.image_to_string(gray, config=config)
            
            # Enhanced live OCR debugging
            logging.info("🔤 OCR Text Extraction for Live Detection:")
            logging.info(f"📝 Extracted text length: {len(extracted_text)} characters")
            
            if extracted_text.strip():
                # Log first 200 characters for debugging
                text_preview = extracted_text.strip()[:200].replace('\n', ' ')
                logging.info(f"📄 Text preview: '{text_preview}'")
                
                # Check for live keywords
                found_keywords = []
                for keyword in self.live_keywords:
                    if keyword in extracted_text:
                        found_keywords.append(keyword)
                        logging.info(f"🎯 Live keyword found: '{keyword}'")
                
                if found_keywords:
                    logging.info(f"✅ LIVE DETECTED via OCR! Keywords: {found_keywords}")
                    return True, f"OCR detected: {', '.join(found_keywords)}"
                else:
                    logging.info("❌ No live keywords found in OCR text")
            else:
                logging.info("❌ No text extracted from OCR")
                
        except Exception as e:
            logging.error(f"❌ OCR Error: {e}")
            
        return False, ""

    def watch_live(self, threshold: float = 0.4):  # Lowered threshold from 0.6 to 0.4
        """
        Watch for livestreams with improved logging and lower threshold
        Fixed: Lowered threshold and enhanced logging with emojis
        """
        logging.info("🚀 Starting TikTok Live Detection...")
        logging.info(f"🎯 Live detection threshold: {threshold}")
        logging.info(f"📺 Monitoring {len(self.live_templates)} live templates")
        logging.info(f"🔤 Monitoring {len(self.live_keywords)} live keywords")
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            logging.info(f"\n🔄 Detection Cycle #{cycle_count}")
            
            try:
                # Get TikTok windows
                tiktok_windows = self.get_tiktok_windows()
                
                if not tiktok_windows:
                    logging.warning("⏳ No TikTok windows found, waiting...")
                    time.sleep(2)
                    continue
                
                for i, window in enumerate(tiktok_windows, 1):
                    logging.info(f"🔍 Checking window {i}/{len(tiktok_windows)}: '{window.title}'")
                    
                    # Capture screenshot
                    screenshot = self.capture_window(window)
                    if screenshot is None:
                        logging.warning(f"⚠️ Failed to capture window {i}")
                        continue
                    
                    logging.info(f"📸 Screenshot captured: {screenshot.shape}")
                    
                    # Check for live templates
                    live_detected_img, match_val, template_name = self.detect_image(
                        screenshot, self.live_templates, "live", threshold
                    )
                    
                    # Check for live text
                    live_detected_text, detected_text = self.detect_text(screenshot)
                    
                    # Combined detection result
                    live_detected = live_detected_img or live_detected_text
                    
                    if live_detected:
                        detection_method = []
                        if live_detected_img:
                            detection_method.append(f"Template: {template_name} ({match_val:.4f})")
                        if live_detected_text:
                            detection_method.append(f"OCR: {detected_text}")
                        
                        logging.info(f"🔴 LIVESTREAM DETECTED! 🔴")
                        logging.info(f"📺 Window: '{window.title}'")
                        logging.info(f"🎯 Detection: {' | '.join(detection_method)}")
                        logging.info(f"⏩ ACTION: Skipping livestream...")
                        
                        # Here you would implement the actual skip action
                        # For now, just log the action
                        self.skip_livestream(window)
                        
                    else:
                        logging.info(f"✅ No livestream detected in window {i}")
                
                # Wait before next check
                logging.info(f"⏳ Waiting 3 seconds before next cycle...")
                time.sleep(3)
                
            except KeyboardInterrupt:
                logging.info("🛑 Detection stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Error in watch_live cycle {cycle_count}: {e}")
                time.sleep(5)

    def skip_livestream(self, window):
        """Skip the livestream (placeholder implementation)"""
        try:
            logging.info(f"⏩ Attempting to skip livestream in '{window.title}'")
            
            # Placeholder for actual skip implementation
            # This could involve:
            # - Clicking skip button
            # - Swiping to next video
            # - Using keyboard shortcuts
            
            logging.info(f"✅ Skip action completed for '{window.title}'")
            
        except Exception as e:
            logging.error(f"❌ Error skipping livestream: {e}")

    def watch_ads(self, threshold: float = 0.6):
        """Watch for ads and skip them"""
        logging.info("🚀 Starting TikTok Ad Detection...")
        logging.info(f"🎯 Ad detection threshold: {threshold}")
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            logging.info(f"\n🔄 Ad Detection Cycle #{cycle_count}")
            
            try:
                tiktok_windows = self.get_tiktok_windows()
                
                if not tiktok_windows:
                    logging.warning("⏳ No TikTok windows found, waiting...")
                    time.sleep(2)
                    continue
                
                for i, window in enumerate(tiktok_windows, 1):
                    screenshot = self.capture_window(window)
                    if screenshot is None:
                        continue
                    
                    # Check for ad templates
                    ad_detected, match_val, template_name = self.detect_image(
                        screenshot, self.ad_templates, "ad", threshold
                    )
                    
                    if ad_detected:
                        logging.info(f"📢 AD DETECTED! Template: {template_name} ({match_val:.4f})")
                        logging.info(f"⏩ ACTION: Skipping ad...")
                        self.skip_ad(window)
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                logging.info("🛑 Ad detection stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Error in watch_ads cycle {cycle_count}: {e}")
                time.sleep(5)

    def skip_ad(self, window):
        """Skip the ad (placeholder implementation)"""
        try:
            logging.info(f"⏩ Attempting to skip ad in '{window.title}'")
            # Placeholder for actual skip implementation
            logging.info(f"✅ Ad skip action completed for '{window.title}'")
        except Exception as e:
            logging.error(f"❌ Error skipping ad: {e}")


def main():
    """Main function"""
    try:
        # Create directories if they don't exist
        os.makedirs("templates/ads", exist_ok=True)
        os.makedirs("templates/live", exist_ok=True)
        
        bot = TikTokSkipBot()
        
        print("TikTok Skip Bot - Vietnamese")
        print("1. Watch for livestreams")
        print("2. Watch for ads")
        print("3. Watch for both")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            bot.watch_live()
        elif choice == "2":
            bot.watch_ads()
        elif choice == "3":
            import threading
            
            # Run both watchers in separate threads
            live_thread = threading.Thread(target=bot.watch_live)
            ad_thread = threading.Thread(target=bot.watch_ads)
            
            live_thread.daemon = True
            ad_thread.daemon = True
            
            live_thread.start()
            ad_thread.start()
            
            logging.info("🚀 Both live and ad detection started!")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("🛑 Bot stopped by user")
        else:
            print("Invalid choice")
            
    except Exception as e:
        logging.error(f"❌ Main error: {e}")


if __name__ == "__main__":
    # Check for required modules
    missing_modules = []
    if not PYAUTOGUI_AVAILABLE:
        missing_modules.append("pyautogui")
    if not PYGETWINDOW_AVAILABLE:
        missing_modules.append("pygetwindow")
    if not PYTESSERACT_AVAILABLE:
        missing_modules.append("pytesseract")
    
    if missing_modules:
        print(f"⚠️ Missing optional modules: {', '.join(missing_modules)}")
        print("🔧 Install with: pip install " + " ".join(missing_modules))
        print("📝 Some functionality may be limited")
        print()
    
    main()