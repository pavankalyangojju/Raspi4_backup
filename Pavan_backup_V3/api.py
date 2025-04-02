from flask import Flask, request, jsonify, render_template_string
import csv
import os
from datetime import datetime

app = Flask(__name__)

CSV_FILE = "attendance_log.csv"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2F4F4F; }
        .container { display: flex; }
        .left { flex: 1; margin-right: 20px; }
        .right { flex: 1; }
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px 12px; border: 1px solid #ccc; text-align: left; }
        th { background-color: #f2f2f2; }
        form { margin-bottom: 20px; }
        input, select { padding: 5px; font-size: 16px; margin-right: 10px; }
        canvas { width: 100% !important; height: auto !important; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Attendance Dashboard</h1>
    <form method="get">
        <label for="date">Select Date:</label>
        <select name="date">
            {% for d in dates %}
            <option value="{{ d }}" {% if selected == d %}selected{% endif %}>{{ d }}</option>
            {% endfor %}
        </select>
        <label for="name">Search Name:</label>
        <input type="text" name="name" value="{{ name or '' }}" placeholder="Enter name"/>
        <input type="submit" value="Filter"/>
    </form>

    <div class="container">
        <div class="left">
            {% if records %}
            <table>
                <thead>
                    <tr><th>Name</th><th>Date</th><th>Time</th></tr>
                </thead>
                <tbody>
                    {% for row in records %}
                    <tr><td>{{ row[0] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td></tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
                <p>No records found for the given search.</p>
            {% endif %}
        </div>
        <div class="right">
            <canvas id="attendanceChart"></canvas>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('attendanceChart').getContext('2d');
        const data = {
            labels: {{ graph_labels | safe }},
            datasets: [{
                label: 'Attendance Count',
                data: {{ graph_values | safe }},
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        };
        const config = {
            type: 'bar',
            data: data,
            options: {
                scales: {
                    y: { beginAtZero: true }
                }
            }
        };
        new Chart(ctx, config);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    records = []
    dates = set()
    name_counts = {}

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) == 4:
                    dates.add(row[2])
                    records.append(row)

    dates = sorted(dates)
    selected_date = request.args.get("date") or (dates[-1] if dates else None)
    search_name = request.args.get("name", "").strip().lower()

    # Filter records by selected date and optional name search
    filtered = [r for r in records if r[2] == selected_date and (not search_name or search_name in r[0].lower())]

    # Count name occurrences from filtered results only
    for r in filtered:
        name = r[0]
        name_counts[name] = name_counts.get(name, 0) + 1

    graph_labels = list(name_counts.keys())
    graph_values = list(name_counts.values())

    return render_template_string(
        HTML_TEMPLATE,
        records=filtered,
        dates=dates,
        selected=selected_date,
        name=search_name,
        graph_labels=graph_labels,
        graph_values=graph_values
    )

@app.route('/attendance', methods=['POST'])
def receive_attendance():
    data = request.json
    name = data.get("name")
    rfid = data.get("rfid")
    datetime_str = data.get("datetime")

    if not all([name, rfid, datetime_str]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        date = dt.strftime("%Y-%m-%d")
        time_ = dt.strftime("%H:%M:%S")
    except ValueError:
        return jsonify({"error": "Invalid datetime format"}), 400

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "RFID", "Date", "Time"])
        writer.writerow([name, rfid, date, time_])

    return jsonify({"message": "Attendance saved successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
