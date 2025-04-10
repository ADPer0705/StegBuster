# StegBuster User Guide

This guide provides detailed instructions on how to use StegBuster for detecting and extracting hidden data from various file types.

## Basic Usage

StegBuster is primarily a command-line tool. The basic syntax is:

```bash
python stegbuster.py [file_to_analyze] [options]
```

### Quick Start Examples

```bash
# Basic analysis of an image
python stegbuster.py suspicious_image.png

# Save extracted data to a file
python stegbuster.py hidden_message.wav -o extracted_data.txt

# Use all detection methods
python stegbuster.py steganography_sample.jpg -a
```

## Command-Line Options

StegBuster provides various options to customize the detection and extraction process:

### Basic Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Save extracted data to specified file |
| `-v, --verbose` | Enable detailed output |
| `-a, --all-methods` | Try all possible extraction methods (slower but more thorough) |
| `-c, --chi-square` | Perform detailed chi-square analysis |
| `-r, --rs-analysis` | Perform detailed RS (Regular-Singular) analysis |
| `--detailed-analysis` | Perform comprehensive steganalysis with all methods |
| `--json` | Output results in JSON format (for detailed analysis) |

### Machine Learning Options

| Option | Description |
|--------|-------------|
| `--use-ml` | Use machine learning for enhanced detection |
| `--smart-ml` | Use ML intelligently, only when needed for better performance |
| `--ml-confidence FLOAT` | Minimum confidence threshold for ML detection (0-1) |
| `--visualize` | Generate visualization of detection results |
| `--advanced-model` | Use the advanced ML model (larger, but more accurate) |

## Working with Different File Types

### Image Files

StegBuster can detect and extract hidden data from various image formats (PNG, JPG, BMP, GIF, etc.):

```bash
# Basic LSB detection
python stegbuster.py image.png

# Detailed analysis with chi-square test
python stegbuster.py image.jpg -c

# ML-based detection with visualization
python stegbuster.py image.bmp --use-ml --visualize
```

#### Example: Detailed Image Analysis

```bash
# Comprehensive analysis with visual output
python stegbuster.py suspicious.png --detailed-analysis --visualize

# Extract with all methods and save to file
python stegbuster.py suspicious.png -a -o extracted.txt
```

### Audio Files

StegBuster can detect steganography in WAV, MP3, and other audio formats:

```bash
# Basic detection
python stegbuster.py audio.wav

# Using all available methods
python stegbuster.py audio.mp3 -a
```

### Text Files

For analyzing text documents with hidden data:

```bash
# Basic analysis
python stegbuster.py document.txt

# With machine learning enhancement
python stegbuster.py document.txt --use-ml
```

## Advanced Features

### Detailed Statistical Analysis

For more comprehensive steganalysis:

```bash
# Perform detailed analysis with chi-square and RS analysis
python stegbuster.py image.png --detailed-analysis

# Output results in JSON format for programmatic usage
python stegbuster.py image.png --detailed-analysis --json
```

### Machine Learning Integration

StegBuster provides ML-based detection that can identify more sophisticated steganography techniques:

```bash
# Basic ML detection
python stegbuster.py image.png --use-ml

# Smart ML that balances performance and accuracy
python stegbuster.py image.png --smart-ml

# Use advanced models for higher accuracy
python stegbuster.py image.png --use-ml --advanced-model

# Adjust confidence threshold
python stegbuster.py image.png --use-ml --ml-confidence 0.85
```

### Visualization

Generate visual representations of detected steganography:

```bash
# Generate visualization for an image
python stegbuster.py image.png --visualize

# Combine with ML for enhanced visualization
python stegbuster.py image.png --use-ml --visualize
```

The visualization will be saved as `[original_filename]_stego_analysis.png` and shows:
1. The original image
2. A heatmap showing potential areas with hidden data
3. LSB plane visualization

## Understanding Results

### Detection Output

When StegBuster detects steganography, it outputs:

```
Steganography detected using [METHOD]. Extracting hidden data...
Hidden Message: [EXTRACTED_MESSAGE]
```

Where `[METHOD]` is the detection method (e.g., LSB, DCT) and `[EXTRACTED_MESSAGE]` is the recovered hidden data.

### Confidence Scores

With ML detection, StegBuster provides confidence scores:

```
=== ML Steganalysis Results ===
Steganography detected: Yes
Confidence: 92.5%
Suspected method: LSB
```

Higher confidence scores (>90%) indicate stronger evidence of steganography.

### JSON Output

When using the `--json` option with detailed analysis, StegBuster outputs structured data:

```json
{
  "file": "suspicious.png",
  "analysis": {
    "Red": {
      "overall": {
        "is_stego_likely": true,
        "confidence": 0.85,
        "detection_votes": 2
      },
      "chi_square": {
        "p_value": 0.002,
        "is_stego_likely": true
      },
      "rs_analysis": {
        "embedding_rate": 0.32,
        "rm_ratio": 0.95,
        "sm_ratio": 0.75,
        "is_stego_likely": true
      }
    },
    "Green": {
      // Similar structure
    },
    "Blue": {
      // Similar structure
    },
    "combined": {
      "is_stego_likely": true,
      "confidence": 0.83
    }
  }
}
```

## Practical Examples

### Example 1: Basic Detection and Extraction

```bash
python stegbuster.py suspicious_image.png
```

Output:
```
Analyzing suspicious_image.png (detected as: image)
Steganography detected using LSB. Extracting hidden data...
Hidden Message: This is a secret message hidden in the image.
```

### Example 2: Using Machine Learning with Visualization

```bash
python stegbuster.py complex_stego.jpg --smart-ml --visualize
```

Output:
```
Analyzing complex_stego.jpg (detected as: image)
Using smart detector with intelligent ML integration
Steganography detected using DCT (confidence: 94.2%) using ML analysis. Extracting hidden data...
Hidden Message: [Binary data of 1024 bytes]
Visualization saved to: complex_stego.jpg_stego_analysis.png
```

### Example 3: Detailed Analysis with JSON Output

```bash
python stegbuster.py suspicious.png --detailed-analysis --json
```

This will generate a comprehensive JSON report with statistical metrics for each color channel.

## Integration with Other Tools

### Automation Scripts

You can easily integrate StegBuster into your automation workflows:

```bash
#!/bin/bash
# Example script to scan multiple files

for file in ./samples/*; do
  echo "Analyzing $file..."
  python stegbuster.py "$file" --json > "${file}_analysis.json"
done
```

### Combining with Other Forensics Tools

StegBuster works well alongside other digital forensics tools:

```bash
# Extract embedded data and pass to other tools
python stegbuster.py suspicious.jpg -o extracted.bin
binwalk extracted.bin
strings extracted.bin | grep -i password
```

## Troubleshooting

### Common Issues

1. **No hidden data found**
   - Try using all methods: `python stegbuster.py file -a`
   - Try ML detection: `python stegbuster.py file --use-ml --advanced-model`

2. **Extraction fails**
   - Some steganography methods require specific knowledge of parameters used during hiding
   - Try with verbose output: `python stegbuster.py file -v`

3. **Binary data output**
   - If the extracted data appears as binary, it might not be text
   - Save to a file and analyze with hex editors or other tools

### Performance Issues

1. **Slow analysis with ML**
   - Use `--smart-ml` instead of `--use-ml` for better performance
   - Avoid `--advanced-model` for routine scans

2. **Memory errors**
   - Large files might cause memory issues
   - Process files in smaller chunks or use a system with more RAM

## Best Practices

1. **Start with basic analysis** before using detailed or ML-based methods
2. **Save extracted data** to files with `-o` for further analysis
3. **Use `--detailed-analysis` with `--json`** for thorough documentation
4. **Combine traditional and ML methods** for best results
5. **Use visualizations** to locate steganography in complex images

## Support and Further Help

For more information or assistance:
- Check the [GitHub repository](https://github.com/yourusername/stegbuster)
- Refer to the [ML Integration Strategy](ml_integration_strategy.md) document
- Consult the [Installation Guide](installation_guide.md) for dependency issues