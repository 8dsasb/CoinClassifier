# AI-Powered Coin Classifier

A machine-learning system for fine-grained classification of coins from 8 countries and 48 denominations.

## Project Overview

This repository implements a deep-learning-based coin classification pipeline capable of distinguishing coins from eight countries and up to 48 individual classes (country × denomination). The system could potentially be useful for automated teller machines, vending units, currency-exchange kiosks, and assistive tools for visually impaired users.

## Data

- **Sources**: Google Images, numismatic databases and forums, auction sites.
- **Structure**:  
  ```
  CoinImages/
  ├── CountryA/
  │   ├── 1h.jpg
  │   ├── 1t.jpg
  │   └── …
  ├── CountryB/
  │   └── …
  └── …
  ```
  - Folder per country, subfolders per denomination.
  - Filenames annotated with `h` (heads) or `t` (tails).

## Preprocessing

1. Resize all images to 224×224 pixels.
2. Convert to grayscale and detect the coin region using Hough Circle Transform.
3. Crop to the detected circle (fallback to full image if detection fails).
4. Apply median blur to reduce noise.
5. Normalize pixel values to [0, 1].

## Data Augmentation

Applied on-the-fly during training to improve generalization:

- Random rotations (±5°)
- Horizontal flips
- Brightness and contrast jitter
- Color jitter (saturation and contrast variations)

## Model Training

Three phases:

1. Baseline training on pretrained CNNs without modification.
2. Fine-tuning with learning-rate scheduling on the collected images.
3. Augmented fine-tuning incorporating augmented and robot-captured images.

### Architectures Evaluated

| Model             | Phase 1 Test Acc | Phase 2 Test Acc  |
|-------------------|------------------|-------------------|
| AlexNet           | 73.5%            | 73.5%             |
| VGG-16            | 85.2%            | 84.8%             |
| RegNet            | 82.5%            | 78.2%             |
| WideResNet        | 81.6%            | 93.7%             |
| EfficientNetV2-B0 | 85.0%            | 82.0%             |
| ResNet-50         | 88.0%            | 69.0%             |
| ResNet-18         | 74.0%            | 89.0%             |
| GoogLeNet         | 81.0%            | 85.0%             |

- Best performer: WideResNet.

## Evaluation

- Metrics: Accuracy, confusion matrix.
- Analysis: Misclassifications inspected to refine preprocessing and augmentation.

## Usage

1. Clone the repository and set up a virtual environment:
   ```bash
   git clone https://github.com/your-username/coin-classifier.git
   cd coin-classifier
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare your `CoinImages/` directory as described above.
3. Run the training script:
   ```bash
   python train.py --data-dir CoinImages --model wide_resnet
   ```
4. Run the inference script:
   ```bash
   python predict.py --image path/to/coin.jpg --checkpoint checkpoints/wide_resnet.pth
   ```

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- torchvision

See `requirements.txt` for full dependency list.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
