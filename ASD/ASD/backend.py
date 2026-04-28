"""
Separate Backend Service for ASD Detection Models
Allows independent testing of image and tabular models
"""

import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import os
import json
from pathlib import Path

class ASDBackend:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = base_dir
        self._load_models()
    
    def _load_models(self):
        """Load all models and required files"""
        try:
            self.scaler = joblib.load(os.path.join(self.base_dir, "scaler.pkl"))
            self.tabular_model = joblib.load(os.path.join(self.base_dir, "logistic_regression_model.sav"))
            self.image_model = tf.keras.models.load_model(os.path.join(self.base_dir, "asd_cnn_model.keras"))
            
            with open(os.path.join(self.base_dir, "class_names.json"), "r") as f:
                self.class_names = json.load(f)
            
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict_tabular(self, input_data):
        """
        Predict using tabular model
        Args:
            input_data: List of 16 features (a1-a10, age, gender, ethnicity, jaundice, family_asd, who)
        Returns:
            dict with prediction and score
        """
        try:
            if len(input_data) != 16:
                return {"error": f"Expected 16 features, got {len(input_data)}"}
            
            input_array = np.asarray(input_data).reshape(1, -1)
            scaled = self.scaler.transform(input_array)
            prediction = self.tabular_model.predict(scaled)[0]
            
            label = "ASD" if prediction == 1 else "Non-ASD"
            score = 1.0 if prediction == 1 else 0.0
            
            return {
                "success": True,
                "prediction": int(prediction),
                "label": label,
                "score": float(score)
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def predict_image(self, image_path):
        """
        Predict using image model
        Args:
            image_path: Path to image file
        Returns:
            dict with prediction and confidence
        """
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}", "success": False}
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            
            # ✅ NORMALIZE TO [0, 1] - FIX FROM EARLIER ANALYSIS
            img_array = img_array / 255.0
            
            # Predict
            predictions = self.image_model.predict(img_array)
            scores = tf.nn.softmax(predictions[0]).numpy()
            
            predicted_class = np.argmax(scores)
            confidence = float(np.max(scores))
            label = self.class_names[predicted_class]
            
            return {
                "success": True,
                "prediction": int(predicted_class),
                "label": label,
                "confidence": confidence,
                "all_scores": {
                    self.class_names[i]: float(scores[i]) 
                    for i in range(len(self.class_names))
                }
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def predict_combined(self, tabular_data, image_path):
        """
        Combined prediction using both models
        Args:
            tabular_data: List of 16 tabular features
            image_path: Path to image file
        Returns:
            dict with combined result
        """
        tabular_result = self.predict_tabular(tabular_data)
        image_result = self.predict_image(image_path)
        
        if not tabular_result.get("success") or not image_result.get("success"):
            return {
                "success": False,
                "error": "Failed to get predictions",
                "tabular": tabular_result,
                "image": image_result
            }
        
        # Combine scores
        combined_score = (tabular_result["score"] + image_result["confidence"]) / 2
        final_label = "ASD" if combined_score >= 0.5 else "Non-ASD"
        
        return {
            "success": True,
            "tabular": tabular_result,
            "image": image_result,
            "combined_score": float(combined_score),
            "final_label": final_label
        }


# Test function
if __name__ == "__main__":
    backend = ASDBackend()
    
    # Test data
    test_tabular = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 36, 1, 0, 1, 0, 0]
    test_image = r"C:\Users\HP\OneDrive\Desktop\ASD\ASD\AutismDataset\test\Test1.jpg"
    
    print("\n" + "="*50)
    print("TESTING TABULAR MODEL")
    print("="*50)
    result = backend.predict_tabular(test_tabular)
    print(result)
    
    print("\n" + "="*50)
    print("TESTING IMAGE MODEL")
    print("="*50)
    if os.path.exists(test_image):
        result = backend.predict_image(test_image)
        print(result)
    else:
        print(f"Image not found: {test_image}")
    
    print("\n" + "="*50)
    print("TESTING COMBINED PREDICTION")
    print("="*50)
    if os.path.exists(test_image):
        result = backend.predict_combined(test_tabular, test_image)
        print(json.dumps(result, indent=2))
