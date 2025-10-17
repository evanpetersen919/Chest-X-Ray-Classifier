# Sample Chest X-Ray Images

This folder contains sample chest X-ray images for testing the classifier application.

## Sample Images

| Filename | Disease Labels | Description |
|----------|----------------|-------------|
| `sample_1_no_finding.png` | No Finding | Normal chest X-ray |
| `sample_2_pneumothorax.png` | Pneumothorax | Collapsed lung |
| `sample_3_effusion_pneumothorax.png` | Effusion, Pneumothorax | Multiple conditions |
| `sample_4_mass.png` | Mass | Lung mass/tumor |
| `sample_5_atelectasis_infiltration.png` | Atelectasis, Infiltration | Collapsed lung + infiltration |

## Usage

### In Streamlit App
1. Run `streamlit run app.py`
2. Upload any image from this folder
3. View the model's predictions

### Quick Test
```bash
streamlit run app.py
```
Then drag and drop one of these sample images into the file uploader.

## Source

All images are from the **NIH Clinical Center Chest X-Ray Dataset**, which is publicly available and free to use for research and educational purposes.

**Dataset Citation:**
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017.

**Dataset Link:** https://nihcc.app.box.com/v/ChestXray-NIHCC

## License

These sample images are provided for demonstration and educational purposes only. The original dataset is in the public domain as a work of the U.S. Government (National Institutes of Health).

**Note:** These samples are for testing the model only and should not be used for any clinical diagnosis or medical decision-making.
