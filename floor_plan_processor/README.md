# Floor Plan Processor

A simplified, portable module for processing architectural floor plans from PDF files.

## Features

- PDF to image conversion with configurable DPI
- Text detection and removal using OCR
- Building outline extraction
- Room detection within building outlines
- Geometry optimization with vertex snapping
- Configurable via workflow.json

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from floor_plan_processor import FloorPlanProcessor

# Initialize with workflow configuration
processor = FloorPlanProcessor('workflow.json')

# Process a PDF file
results = processor.process_pdf('floor_plan.pdf', page_number=0)

# Save results
processor.save_results(results, 'output_directory')
```

### Process Image Directly

```python
import cv2
from floor_plan_processor import FloorPlanProcessor

processor = FloorPlanProcessor('workflow.json')
image = cv2.imread('floor_plan.png')
results = processor.process_image(image)
```

### Results Structure

The processor returns a dictionary containing:
- `original_image`: Original rendered image
- `processed_image`: Preprocessed binary image
- `building_outline`: Main building contour
- `rooms`: List of room contours
- `optimized_rooms`: Geometrically optimized room contours
- `visualization`: Visualization image

## Workflow Configuration

The `workflow.json` file contains all processing parameters. Key parameters include:

- `render_dpi`: PDF rendering resolution (default: 600)
- `thresh_method`: Thresholding method (6 = Otsu)
- `remove_text`: Enable text removal via OCR
- `extract_rooms`: Enable room extraction
- `min_room_area`: Minimum room area in pixels
- `max_room_area`: Maximum room area in pixels

## Integration

To integrate into your project:

1. Copy the `floor_plan_processor` directory
2. Copy `workflow.json` to your project
3. Install requirements
4. Import and use as shown above

## Debug Output

When processing rooms, debug images are saved to `debug_room_extraction/` directory showing each processing step.