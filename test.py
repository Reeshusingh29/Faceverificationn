from flask import Flask, request, render_template_string
import cv2
import numpy as np
import os

app = Flask(__name__)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image cannot be read.")
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

def compute_hog_features(img):
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

    cell_size = 8
    bins = 9
    hog_vector = []

    for i in range(0, img.shape[0], cell_size):
        for j in range(0, img.shape[1], cell_size):
            mag_cell = mag[i:i+cell_size, j:j+cell_size]
            angle_cell = angle[i:i+cell_size, j:j+cell_size]
            hist = np.zeros(bins)

            for m in range(cell_size):
                for n in range(cell_size):
                    bin_idx = int(angle_cell[m, n] / (180 / bins)) % bins
                    hist[bin_idx] += mag_cell[m, n]
            hog_vector.extend(hist)
    hog_vector = np.array(hog_vector)
    if np.linalg.norm(hog_vector) != 0:
        hog_vector = hog_vector / np.linalg.norm(hog_vector)
    return hog_vector

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
<title>Face Similarity Check</title>
</head>
<body>
  <h2>Upload Your Aadhaar Image and Selfie</h2>
  <form method="POST" enctype="multipart/form-data">
    <label>Aadhaar Image:</label><br>
    <input type="file" name="aadhaar" accept="image/*" required><br><br>
    <label>Selfie Image:</label><br>
    <input type="file" name="selfie" accept="image/*" required><br><br>
    <button type="submit">Check Similarity</button>
  </form>

  {% if similarity is not none %}
  <h3>Similarity Score: {{ similarity }}%</h3>
  {% endif %}

  {% if error %}
  <p style="color:red;">Error: {{ error }}</p>
  {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    error = None
    if request.method == 'POST':
        try:
            aadhaar_file = request.files['aadhaar']
            selfie_file = request.files['selfie']
            
            aadhaar_path = "aadhaar_temp.jpg"
            selfie_path = "selfie_temp.jpg"
            aadhaar_file.save(aadhaar_path)
            selfie_file.save(selfie_path)

            aadhaar_img = preprocess_image(aadhaar_path)
            selfie_img = preprocess_image(selfie_path)

            score = cosine_similarity(compute_hog_features(aadhaar_img), compute_hog_features(selfie_img))
            similarity = round(score * 100, 2)

            # Clean temp files
            os.remove(aadhaar_path)
            os.remove(selfie_path)
        except Exception as e:
            error = str(e)

    return render_template_string(HTML_PAGE, similarity=similarity, error=error)

if __name__ == '__main__':
    app.run(debug=True)
