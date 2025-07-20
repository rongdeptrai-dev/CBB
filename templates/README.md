# TikTok Skip Bot Test Templates

This directory contains template images for detection:

## ads/ directory
Contains templates for advertisement detection
- Put PNG/JPG images of ad elements here
- Templates will be used for template matching

## live/ directory  
Contains templates for livestream detection
- Put PNG/JPG images of live streaming elements here
- Common elements: LIVE badges, streaming indicators, etc.

## Usage
1. Add template images to appropriate directories
2. Run khovl.py to start detection
3. Check logs for template matching results

## Template Naming
Use descriptive names:
- ads/skip_button.png
- ads/ad_banner.png  
- live/live_badge.png
- live/streaming_indicator.png