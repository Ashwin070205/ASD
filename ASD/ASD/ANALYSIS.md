# Image Model Discrepancies Analysis

## Key Differences Between Combined.ipynb and app.py

### 1. **Image Loading Method** ❌ DIFFERENT
**Combined.ipynb:**
```python
img = tf.keras.utils.load_img(
    r"C:\Users\HP\OneDrive\Desktop\ASD\ASD\AutismDataset\test\Test1.jpg",
    target_size=(224, 224)
)
img_array = tf.keras.utils.img_to_array(img)
```

**app.py:**
```python
image = Image.open(uploaded_image).resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(image)
```

**Issue:** 
- `tf.keras.utils.load_img()` automatically applies **RGB scaling** (normalizes pixel values)
- `PIL Image.open()` loads raw pixel values **without normalization**
- This causes different input ranges to the model

### 2. **Probability Conversion** ✓ SAME
Both use softmax (though slightly different):

**Combined.ipynb:**
```python
score = tf.nn.softmax(predictions[0])
```

**app.py:**
```python
image_score = tf.nn.softmax(preds[0]).numpy()
```

Both are equivalent - just app.py converts to numpy.

---

## **ROOT CAUSE: Pixel Value Normalization**

### Combined.ipynb flow:
1. `tf.keras.utils.load_img()` → Loads and **auto-normalizes to [0, 1]**
2. `img_to_array()` → Converts to array
3. Model receives normalized values ✓

### app.py flow:
1. `Image.open()` → Loads with **raw pixel values [0, 255]**
2. `.resize()` → Resizes but doesn't normalize
3. `img_to_array()` → Converts to array with **[0, 255] range**
4. Model receives non-normalized values ❌

---

## **SOLUTION**

Add normalization in app.py after img_array creation:

```python
# Before: img_array with values [0, 255]
img_array = tf.keras.preprocessing.image.img_to_array(image)
img_array = np.expand_dims(img_array, axis=0)

# ADD THIS LINE:
img_array = img_array / 255.0  # Normalize to [0, 1]

# Now predict
preds = image_model.predict(img_array)
```

---

## **Expected Impact**

After normalization:
- ✅ app.py predictions will match Combined.ipynb
- ✅ Model accuracy will improve
- ✅ Consistent results across both implementations
