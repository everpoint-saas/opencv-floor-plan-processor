# OpenCV Floor Plan Processor

Computer vision-based architectural floor plan analysis tool using OpenCV. Automatically extracts building outlines, detects rooms, and optimizes geometry from PDF floor plans.

## Features

- **PDF to Image Conversion**: High-resolution rendering with configurable DPI
- **Text Detection & Removal**: OCR-based text cleanup for cleaner contour detection
- **Building Outline Extraction**: Automatic detection of main building boundaries
- **Room Detection**: Identifies individual rooms within building outlines
- **Geometry Optimization**: Vertex snapping and geometric cleanup
- **Interactive UI**: PyQt-based interface for visual processing
- **Configurable Workflow**: JSON-based configuration for all processing parameters

## Project Structure

```
opencv-floor-plan-processor/
├── floor_plan_processor/    # Modern processing library (recommended)
│   ├── processor.py         # Main processor class
│   ├── contour_extractor.py # Building outline extraction
│   ├── room_extractor.py    # Room detection
│   ├── geometry_optimizer.py # Geometric optimization
│   ├── text_remover.py      # OCR-based text removal
│   ├── pdf_handler.py       # PDF rendering
│   ├── example.py           # Usage examples
│   └── workflow.json        # Configuration file
├── ui/                      # UI components
│   ├── main_window.py       # Main application window
│   ├── tabs.py              # Parameter tabs
│   └── handlers.py          # Event handlers
├── core/                    # Core processing modules
│   ├── preprocess.py        # Image preprocessing
│   ├── functional.py        # Functional pipeline
│   ├── geometry_optimizer.py
│   ├── image_processor.py
│   ├── pdf_loader.py
│   └── ocr_handler.py
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8+
- OpenCV

### Install Dependencies

```bash
pip install -r requirements.txt
```

EasyOCR will be automatically installed with the dependencies. On first run, it will download the required language models.

## Usage

### 1. Using the GUI Application

```bash
python main.py
```

Features:
- Load PDF floor plans
- Select processing page
- Configure processing parameters
- Visualize results in real-time
- Export processed data

### 2. Using the Library (Recommended)

```python
from floor_plan_processor import FloorPlanProcessor

# Initialize with workflow configuration
processor = FloorPlanProcessor('floor_plan_processor/workflow.json')

# Process a PDF file
results = processor.process_pdf('your_floor_plan.pdf', page_number=0)

# Access results
building_outline = results['building_outline']
rooms = results['optimized_rooms']
```

### 3. Using Core Modules Directly

```python
from core.preprocess import Preprocessor
from core.functional import ContourExtractor

# Use individual components
preprocessor = Preprocessor()
preprocessor.load_pdf('floor_plan.pdf')
image = preprocessor.get_page_image(0)
```

## Configuration

Edit `floor_plan_processor/workflow.json` to customize processing:

```json
{
  "render_dpi": 600,           # PDF rendering resolution
  "thresh_method": 6,          # Thresholding method (6=Otsu)
  "remove_text": true,         # Enable OCR text removal
  "extract_rooms": true,       # Enable room detection
  "min_room_area": 5000,      # Minimum room size (pixels)
  "max_room_area": 2000000,   # Maximum room size (pixels)
  "snap_threshold": 10         # Vertex snapping distance
}
```

## Output

The processor generates:

- **Processed Image**: Binary threshold image with text removed
- **Building Outline**: Main building contour
- **Room Contours**: Individual room boundaries
- **Optimized Geometry**: Cleaned-up room shapes
- **Debug Visualizations**: Step-by-step processing images (saved to `debug_room_extraction/`)

## Use Cases

- **LEED Certification**: Automated floor area calculations
- **Space Planning**: Room layout analysis
- **Building Analysis**: Geometric property extraction
- **Architectural Documentation**: Digital plan processing

## Development

### Running Tests

```python
# Process your own floor plan file
python floor_plan_processor/example.py
```

Note: Example input files are not included due to IP restrictions. Use your own PDF floor plans for testing.

### Debug Mode

Debug images are automatically saved to `debug_room_extraction/` showing:
- Threshold processing
- Contour detection stages
- Room extraction steps
- Geometry optimization results

## Dependencies

Core dependencies (see `requirements.txt`):
- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `easyocr`: OCR engine (with Korean/English support)
- `PyMuPDF (fitz)`: PDF rendering
- `PyQt6`: UI framework (for main_ui.py)

## License

MIT License - See LICENSE file for details

## Author

J.Y. Lee - [Everpoint](https://everpoint.net)

## Related Projects

This tool is part of the VERTIQ LEED automation platform.
