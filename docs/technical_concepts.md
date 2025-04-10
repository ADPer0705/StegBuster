# StegBuster Technical Concepts Reference

This document provides an overview of the technical concepts, algorithms, and research behind the steganography detection methods used in StegBuster.

## Steganography Techniques

### Image Steganography

#### LSB (Least Significant Bit) Steganography

LSB steganography modifies the least significant bits of pixel values to hide data:

```
Original pixel: 10101100 (decimal: 172)
Hidden bit: 1
Modified pixel: 10101101 (decimal: 173)
```

The difference is visually imperceptible but can store significant data:
- An RGB image with dimensions 1024×768 can hide up to 2.36 million bits (294 KB) of data

**Implementation details:**
- In images, LSB steganography works by replacing the least significant bits of color values
- Can be applied to all color channels (RGB) or selected channels
- Various implementation strategies:
  - Sequential: Modifies bits in order
  - Pseudo-random: Uses a seed to select pixels in pseudo-random order
  - Adaptive: Modifies bits in areas with high texture/complexity

#### DCT (Discrete Cosine Transform) Steganography

Used primarily in JPEG images which already store data in DCT coefficients:

1. The image is divided into 8×8 blocks
2. DCT is applied to each block
3. Values are quantized (lossy compression)
4. Secret data is embedded by modifying the quantized DCT coefficients

**Common implementations:**
- JSteg: Replaces LSBs of non-zero DCT coefficients
- F5: Uses matrix embedding to minimize changes
- OutGuess: Preserves statistical properties of DCT coefficients

### Audio Steganography

#### Echo Hiding

Embeds data by introducing small echoes:
- Different echo delays represent binary 0 or 1
- Echo amplitude is kept low enough to remain imperceptible
- Kernel mixing is used to distribute the echo

#### Phase Coding

Modifies the phase of sound segments:
- Audio is divided into segments
- Phase of initial segment is modified to represent hidden data
- Phases of subsequent segments are adjusted to maintain phase continuity
- Works because human hearing is less sensitive to absolute phase

### Text Steganography

#### Zero-Width Characters

Uses invisible/non-printing Unicode characters:
- Zero-width spaces (U+200B)
- Zero-width non-joiners (U+200C)
- Zero-width joiners (U+200D)
- Various combinations can encode bits of data

#### Whitespace Manipulation

- Extra spaces between words (1 space = 0, 2 spaces = 1)
- Extra spaces at end of lines
- Tabs vs spaces for indentation

## Detection Methods

### Statistical Methods

#### Chi-Square Test

Tests if the distribution of LSBs follows expected random distribution:

```python
def chi_square_attack_lsb(data, sample_size=None):
    """
    Implements the chi-square attack for LSB steganography detection
    
    Returns:
        chi2_statistic: The chi-square statistic
        p_value: The p-value (lower values suggest steganography)
        is_stego_likely: Boolean indicator
    """
    # Implementation details
    # ...
```

**Key concept:** If LSBs are used for hiding data, their distribution becomes more uniform than expected in natural images.

#### RS (Regular-Singular) Analysis

Analyzes how "regular" and "singular" groups of pixels change when flipping LSBs:

1. Divide the image into groups of pixels
2. Classify each group as regular, singular, or unusable based on smoothness
3. Flip LSBs and reclassify
4. Compare the counts of regular and singular groups before and after flipping
5. Natural images maintain certain statistical relationships, steganography disrupts these relationships

**Expected results:**
- For clean images: RM ≈ R-M and SM ≈ S-M
- For stego images: These relations are significantly disturbed

#### Sample Pair Analysis

Examines pairs of values to detect LSB embedding:

1. Count pairs of adjacent pixel values (value, value+1)
2. Analyze the distribution of these pairs
3. In natural images, these pairs follow certain patterns
4. LSB steganography disrupts these patterns in detectable ways

### Machine Learning Methods

#### Feature Extraction

Common features used for steganalysis:

1. **Rich Models**: High-dimensional feature sets
   - SPAM (Subtractive Pixel Adjacency Matrix)
   - SRM (Spatial Rich Model)

2. **Deep Learning Features**:
   - Convolutional layer activations
   - Noise residuals
   - Custom steganalysis filters

#### ML Architectures

1. **CNN Models for Image Steganalysis**:
   - YeNet: Specialized first layer with 30 high-pass filters
   - SRNet: Deep residual network designed for steganalysis
   - XuNet: Custom architecture with noise filtering front-end

2. **Audio ML Detection**:
   - 1D CNNs for raw audio samples
   - 2D CNNs for spectrograms
   - RNNs/LSTMs for temporal analysis

## Advanced Statistical Concepts

### Markov Chain Features

Captures pixel value dependencies using transition probability matrices:
- Models how pixel values change in relation to neighbors
- Steganography often disrupts these natural patterns
- First-order or higher-order models can be used

### Co-occurrence Matrices

Analyzes joint probability distributions of neighboring pixels:
1. Calculate pixel differences in multiple directions
2. Form co-occurrence matrices from these differences
3. Extract statistical moments as features

## Integration with Machine Learning

### Hybrid Detection Approach

StegBuster integrates statistical and ML methods:

1. **Initial Screening**: Fast statistical tests to identify suspicious files
2. **Detailed Analysis**: More computation-intensive statistical methods
3. **ML Confirmation**: ML models to validate findings and detect sophisticated steganography

### ML Confidence Scoring

Interpreting ML model outputs:

```
Confidence = P(Stego) / (P(Stego) + P(Clean))
```

Where:
- P(Stego) is the model's predicted probability of steganography
- P(Clean) is the model's predicted probability of clean content

### Detection Calibration

Optimal threshold selection based on:
- False positive rate (FPR)
- False negative rate (FNR)
- Equal error rate (EER) where FPR = FNR

## Research References

Key research papers informing our implementation:

1. Fridrich, J., Goljan, M., & Du, R. (2001). "Reliable detection of LSB steganography in color and grayscale images." *Proceedings of the ACM Workshop on Multimedia and Security*, 27-30.

2. Dumitrescu, S., Wu, X., & Wang, Z. (2003). "Detection of LSB steganography via sample pair analysis." *IEEE Transactions on Signal Processing*, 51(7), 1995-2007.

3. Pevny, T., Bas, P., & Fridrich, J. (2010). "Steganalysis by subtractive pixel adjacency matrix." *IEEE Transactions on Information Forensics and Security*, 5(2), 215-224.

4. Ye, J., Ni, J., & Yi, Y. (2017). "Deep Learning Hierarchical Representations for Image Steganalysis." *IEEE Transactions on Information Forensics and Security*, 12(11), 2545-2557.

## Implementation Best Practices

### Performance vs. Accuracy

Balancing considerations:

1. **Memory constraints**:
   - Process large files in chunks
   - Use feature selection to reduce dimensionality

2. **Processing speed**:
   - Progressive analysis (start with fast methods)
   - Early termination when confident

3. **Detection accuracy**:
   - Ensemble methods combining multiple detectors
   - Weighted voting based on confidence

### Statistical Test Parameters

Optimal parameters based on experimental data:

- **Chi-square test**: Sample size of 10,000 pixels provides good balance
- **RS Analysis**: Block size of 4×4 pixels, mask [0,1,1,0]
- **Sample Pair Analysis**: Neighborhood size of 3×3 for optimal sensitivity