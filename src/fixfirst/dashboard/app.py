import sys
from pathlib import Path
import io

import pandas as pd
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ensure we can import from src
src_path = str(Path(__file__).resolve().parents[3])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fixfirst.dashboard.services import inference_service, analysis_service
from fixfirst.dashboard.components import charts

app = Flask(__name__)
app.secret_key = "fixfirst-dev-key"

RESULTS_STORE = {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        reviews_df = pd.DataFrame()
        
        # Determine input method
        if "text_input" in request.form and request.form["text_input"].strip():
            text_input = request.form["text_input"]
            reviews = [r.strip() for r in text_input.split('\n') if r.strip()]
            reviews_df = pd.DataFrame({"review": reviews})
        elif "csv_upload" in request.files:
            file = request.files["csv_upload"]
            if file.filename != '':
                try:
                    reviews_df = pd.read_csv(file)
                    if "review" not in reviews_df.columns:
                        return render_template("index.html", error="CSV must contain a column named 'review'")
                except Exception as e:
                    return render_template("index.html", error=f"Error reading CSV: {e}")
        
        if reviews_df.empty:
            return render_template("index.html", error="Please provide some reviews to analyze.")
            
        # Run inference
        aspects_df = inference_service.run_dashboard_inference(reviews_df)
        
        # Compute priorities and summary
        priority_df = analysis_service.compute_feature_priorities(aspects_df)
        summary = analysis_service.summarize_results(aspects_df, priority_df)
        
        # Prepare charts
        feature_dist_json = charts.render_feature_distribution(priority_df)
        sentiment_dist_json = charts.render_sentiment_distribution(aspects_df)
        priority_chart_json = charts.render_priority_chart(priority_df)
        
        # Store results for PRG pattern
        result_id = str(uuid.uuid4())
        RESULTS_STORE[result_id] = {
            "aspects": aspects_df.to_dict(orient="records") if not aspects_df.empty else None,
            "priorities": priority_df.to_dict(orient="records") if not priority_df.empty else None,
            "summary": summary,
            "feature_dist_json": feature_dist_json,
            "sentiment_dist_json": sentiment_dist_json,
            "priority_chart_json": priority_chart_json,
            "num_reviews": len(reviews_df)
        }
        
        return redirect(url_for('index', result_id=result_id))
        
    # GET request
    result_id = request.args.get("result_id")
    if result_id and result_id in RESULTS_STORE:
        data = RESULTS_STORE[result_id]
        return render_template("index.html", **data)
        
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
