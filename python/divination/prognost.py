from datetime import datetime
from chinese_divination.qimen import (
    QimenDunjia,
    export_plate_to_html,
)

# Initialize the system
qimen = QimenDunjia()

# Calculate plate for any datetime
plate = qimen.calculate(datetime.now())

# Display the plate
print(plate.format_display())

# Get comprehensive analysis
analysis = qimen.analyze(plate)

print(analysis['overall_assessment']['rating'])

# Spirit analysis
analyzer = qimen.analyzer
spirits = analyzer.analyze_spirits(plate)
print(spirits['auspicious_spirits'])

# Star-gate combinations
combos = analyzer.analyze_star_gate_combinations(plate)
print(combos['best_combinations'])

# Export to HTML file
export_plate_to_html(plate, "my_qimen_reading.html")