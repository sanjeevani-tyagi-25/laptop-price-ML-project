from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("laptop_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:

        # SAFE INPUT FUNCTIONS
        def get_int(name):
            value = request.form.get(name)
            return int(value) if value else 0

        def get_float(name):
            value = request.form.get(name)
            return float(value) if value else 0.0

        # FETCH VALUES
        brand = get_int("brand")
        name = get_int("name")
        processor = get_int("processor")
        cpu = get_int("CPU")

        spec_rating = get_float("spec_rating")

        ram = get_float("Ram")
        ram_type = get_int("Ram_type")

        storage = get_float("ROM")
        storage_type = get_int("ROM_type")

        gpu = get_int("GPU")

        display = get_float("display_size")
        res_w = get_float("resolution_width")
        res_h = get_float("resolution_height")

        os_sys = get_int("OS")
        warranty = get_float("warranty")

        # FEATURE ARRAY (IMPORTANT ORDER)
        features = [[
            brand,
            name,
            spec_rating,
            processor,
            cpu,
            ram,
            ram_type,
            storage,
            storage_type,
            gpu,
            display,
            res_w,
            res_h,
            os_sys,
            warranty
        ]]

        prediction = model.predict(np.array(features))
        output = round(prediction[0], 2)

        # 💰 PRICE CATEGORY
        if output < 40000:
            category = "💰 Budget Laptop"
        elif output < 80000:
            category = "⚡ Mid-Range Laptop"
        else:
            category = "🔥 Premium Laptop"

        # 🖼 BRAND IMAGE MAPPING
        brand_images = {
            0: "dell.png",
            1: "hp.png",
            2: "lenovo.png",
            3: "asus.png",
            4: "acer.png",
            5: "apple.png"
        }

        image = brand_images.get(brand, "default.png")

        return render_template(
            "index.html",
            prediction_text=f"₹ {output}",
            category=category,
            image=image
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
