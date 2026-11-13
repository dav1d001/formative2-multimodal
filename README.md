
# Multimodal User Authentication and Product Recommendation

## Project Overview
This project develops a multimodal authentication system that combines facial recognition and voice verification to authenticate users before providing personalized product recommendations.

The system follows three main steps:
1. **Face Recognition:** Identifies the user using facial features.
2. **Voice Verification:** Confirms the user's identity through voice recognition.
3. **Product Recommendation:** Provides product recommendations after successful authentication.

If authentication fails, access is denied.

## Technologies Used
- Python
- Pandas and NumPy
- OpenCV
- Scikit-learn
- Librosa
- Jupyter Notebook

##Demo
demo video: https://youtu.be/VF_xdi1i7bM

## Project Structure
- `src/` – Source code for data processing, model development and authentication.
- `notebooks/` – Jupyter notebooks for data analysis and experiments.
- `task3_audio/` – Audio recordings and extracted audio features.
- `models/` – Saved trained models.
- `data/` – Project datasets and processed data.

## Installation

Clone the repository:

```bash
git clone https://github.com/dav1d001/formative2-multimodal.git
cd formative2-multimodal
```

Create a virtual environment:

```bash
python -m venv venv
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install librosa
```

## Running the Project

Train the models:

```bash
python src/train_all_models.py
```

Run the authentication demonstration:

```bash
python src/auth_demo.py
```

## Team Members
- Junior Nkuba Igiraneza
- Milena Sheila Kaliza Rwakunda
- Rolande Tumugane
- Oluwadamipe David Akintayo

## Privacy
This project is developed for educational purposes. User data and biometric information should be handled responsibly and securely.

