import pandas as pd
import numpy as np
import random

# --- Define Feature Options (Heavily Adjusted for Price Cap < 50M) ---

# Increase low-end lines, drastically reduce high-end. Cap ALL lines at 50M max.
product_lines = {
    'Inspiron': {'prob': 0.60, 'base_price': 9_800_000, 'max_price': 45_000_000, 'segment': 'Consumer'}, # Allow slightly higher max for variance, but capped later
    'Vostro': {'prob': 0.32, 'base_price': 10_800_000, 'max_price': 48_000_000, 'segment': 'Business'}, # Allow slightly higher max for variance, but capped later
    'Latitude': {'prob': 0.04, 'base_price': 20_000_000, 'max_price': 50_000_000, 'segment': 'Business/Ultrabook'},
    'G Series': {'prob': 0.03, 'base_price': 22_000_000, 'max_price': 50_000_000, 'segment': 'Gaming'},
    'XPS': {'prob': 0.01, 'base_price': 28_000_000, 'max_price': 50_000_000, 'segment': 'Ultrabook/Premium'}, # Very rare
    'Alienware': {'prob': 0.00, 'base_price': 0, 'max_price': 0, 'segment': 'Gaming'} # Effectively removed
}
# Normalize probabilities (excluding Alienware if prob is 0)
active_lines = {k: v for k, v in product_lines.items() if v['prob'] > 0}
total_prob = sum(details['prob'] for details in active_lines.values())
for line in active_lines:
    product_lines[line]['prob'] /= total_prob

segments = ['Consumer', 'Business', 'Gaming', 'Ultrabook', 'Premium'] # Removed Workstation

cpu_vendors = ['Intel', 'AMD']
intel_models = ['Core i3', 'Core i5', 'Core i7', 'Core Ultra 5'] # Removed Ultra 7
amd_models = ['Ryzen 3', 'Ryzen 5', 'Ryzen 7']

cpu_generations_intel = ['11', '12', '13', '14', 'Ultra'] # Keep Ultra for Ultra 5
cpu_generations_amd = ['5000', '6000', '7000']

ram_options = [8, 16] # Strictly 8 or 16 GB
# RAM_Type removed

storage_options = [256, 512, 1000] # Capped at 1TB

gpu_types = ['Integrated', 'Dedicated']

# Combined GPU Names (Reduced options significantly)
gpu_names_intel_integrated = ['Intel UHD Graphics', 'Intel Iris Xe'] # Removed Integrated Arc
gpu_names_amd_integrated = ['AMD Radeon Graphics']

# Very limited dedicated options
gpu_names_nvidia_dedicated = ['NVIDIA MX550', 'NVIDIA RTX 3050'] # Only lowest end
gpu_names_amd_dedicated = ['AMD Radeon RX 6500M']
gpu_names_intel_dedicated = [] # Removed dedicated Arc

all_integrated_gpu_names = gpu_names_intel_integrated + gpu_names_amd_integrated
all_dedicated_gpu_names = gpu_names_nvidia_dedicated + gpu_names_amd_dedicated

gpu_vram_options = {
    'Integrated': [0],
    'Intel UHD Graphics': [0],
    'Intel Iris Xe': [0],
    'AMD Radeon Graphics': [0],
    'NVIDIA MX550': [2],
    'NVIDIA RTX 3050': [4], # Only 4GB variant assumed available at this price point
    'AMD Radeon RX 6500M': [4],
}

# Screen_Size_Inches removed
# Screen_Resolution removed
screen_panels = ['Standard', 'IPS'] # Removed OLED
screen_refresh_rates = [60, 90, 120] # Capped at 120Hz
screen_touch_options = ['Yes', 'No']

battery_options = [41, 54, 60] # Only smaller batteries

# --- Price Calculation Logic (Heavily Adjusted for < 50M Cap) ---
def calculate_price(specs):
    line_details = product_lines[specs['Product_Line']]
    base_price = line_details['base_price']
    price = base_price

    # CPU Contribution (Reduced impact)
    cpu_tier = {
        'Core i3': 1.0, 'Ryzen 3': 1.0,
        'Core i5': 2.2, 'Ryzen 5': 2.2, 'Core Ultra 5': 2.5, # Reduced Ultra 5 impact
        'Core i7': 3.5, 'Ryzen 7': 3.5,
    }
    price += cpu_tier.get(specs['CPU_Model'], 1) * 1_800_000 # Reduced base multiplier

    # CPU Generation bonus (Reduced)
    gen_bonus = 0
    if specs['CPU_Vendor'] == 'Intel':
        gen_map = {'11': 0, '12': 250_000, '13': 500_000, '14': 750_000, 'Ultra': 1_000_000}
        gen_bonus = gen_map.get(specs['CPU_Generation'], 0)
    elif specs['CPU_Vendor'] == 'AMD':
        gen_map = {'5000': 0, '6000': 400_000, '7000': 800_000}
        gen_bonus = gen_map.get(specs['CPU_Generation'], 0)
    price += gen_bonus

    # RAM Contribution (8 vs 16 only)
    ram_multiplier = {8: 0, 16: 1.0} # Reduced 16GB cost impact
    ram_base_cost = 1_000_000
    price += ram_multiplier[specs['RAM_GB']] * ram_base_cost
    # RAM_Type bonus removed

    # Storage Contribution (Reduced impact > 512GB)
    storage_multiplier = {256: 0, 512: 1.0, 1000: 2.5}
    storage_base_cost = 650_000
    price += storage_multiplier[specs['Storage_GB']] * storage_base_cost

    # GPU Contribution (Very limited dedicated impact)
    if specs['GPU_Type'] == 'Dedicated':
        gpu_tier = {
            'NVIDIA MX550': 1.0,
            'AMD Radeon RX 6500M': 1.5,
            'NVIDIA RTX 3050': 2.5,
        }
        gpu_base_cost = 1_400_000 # Reduced base GPU cost
        price += gpu_tier.get(specs['GPU_Name'], 0) * gpu_base_cost
        # VRAM bonus adjusted (only 2GB or 4GB possible)
        vram_bonus = {0: 0, 2: 100_000, 4: 300_000}
        price += vram_bonus.get(specs['GPU_VRAM_GB'], 0)

    # Screen Contribution (Panel, Refresh, Touch only)
    # Screen_Size bonus removed
    # Screen_Resolution bonus removed

    panel_bonus = {'Standard': 0, 'IPS': 300_000} # OLED removed
    price += panel_bonus.get(specs['Screen_Panel_Type'], 0)

    refresh_bonus = {60: 0, 90: 250_000, 120: 500_000} # Reduced high refresh bonus
    price += refresh_bonus.get(specs['Screen_Refresh_Rate_Hz'], 0)

    if specs['Screen_Touch'] == 'Yes':
        price += 600_000 # Reduced touch bonus

    # Battery Contribution (Minor)
    battery_bonus = {41: 0, 54: 150_000, 60: 250_000}
    price += battery_bonus.get(specs['Battery_Whr'], 0)

    # Add noise
    noise = price * random.uniform(-0.08, 0.08) # Slightly increased noise range (+/- 8%)
    price += noise

    # --- Enforce Price Limits ---
    # Floor: 90% of base price
    price = max(base_price * 0.9, price)
    # Ceiling: STRICT 50 Million Cap
    price = min(price, 50_000_000)

    # Stronger pull towards 10-25M for Inspiron/Vostro
    if specs['Product_Line'] in ['Inspiron', 'Vostro']:
        target_min = 10_000_000
        target_max = 25_000_000
        if price < target_min:
             price = random.uniform(target_min, target_min + 2_000_000) # Boost into range
        elif price > target_max:
             # Allow some overshoot, but pull most back
             if random.random() < 0.8: # 80% chance to pull back
                price = random.uniform(target_max - 3_000_000, target_max)
             else: # Allow some to slightly exceed 25M up to calculated value (still capped at 50M)
                price = min(price, 35_000_000) # Add a softer cap here for these lines

    # Round to nearest 10,000 VND
    price = round(price / 10000) * 10000

    # Final absolute check
    price = min(price, 50_000_000)

    return int(price)

# --- Data Generation ---
data = []
num_rows = 2000

while len(data) < num_rows:
    row = {}

    # Select Product Line (Alienware excluded)
    active_lines_list = list(active_lines.keys())
    active_probs = [product_lines[line]['prob'] for line in active_lines_list]
    selected_line = random.choices(active_lines_list, weights=active_probs, k=1)[0]

    line_info = product_lines[selected_line]
    row['Product_Line'] = selected_line
    row['Segment'] = line_info['segment']

    # Model Name Generation (Simplified - No screen size)
    try:
        prefix = selected_line
        if selected_line == "G Series": prefix = "G15" # Assume 15 inch for G Series name
        elif selected_line == "XPS": prefix = "XPS 14" # Assume 14 inch for XPS name

        model_suffix = str(random.randint(1000, 9999))
        if selected_line in ["Inspiron", "Vostro"]: model_suffix = str(random.choice([3,5]) * 1000 + random.randint(100,999)) # Limit to 3000/5000 series
        if selected_line == "Latitude": model_suffix = str(random.choice([3,5,7]) * 1000 + random.randint(10,99)) # Limit to 7000 series max

        row['Model_Name'] = f"{prefix} {model_suffix}"
    except Exception:
        row['Model_Name'] = f"{selected_line} Generic"


    # CPU Vendor (Intel/AMD only)
    cpu_vendor_weights = {'Intel': 0.6, 'AMD': 0.4}
    if selected_line == 'Latitude': cpu_vendor_weights = {'Intel': 0.75, 'AMD': 0.25}
    row['CPU_Vendor'] = random.choices(list(cpu_vendor_weights.keys()), weights=list(cpu_vendor_weights.values()), k=1)[0]

    # CPU Model and Generation (Strictly lower/mid)
    if row['CPU_Vendor'] == 'Intel':
        model_options = intel_models
        if selected_line in ['Inspiron', 'Vostro']: model_weights = [0.55, 0.44, 0.01, 0.0] # Heavy i3/i5, tiny i7 chance
        else: model_weights = [0.1, 0.5, 0.3, 0.1] # Mix i3/i5/i7/Ultra5 for others
        row['CPU_Model'] = random.choices(model_options, weights=model_weights, k=1)[0]
        if 'Ultra' in row['CPU_Model']: row['CPU_Generation'] = 'Ultra'
        else: row['CPU_Generation'] = random.choices(['12', '13', '14'], weights=[0.3, 0.4, 0.3], k=1)[0] # Balanced gens

    elif row['CPU_Vendor'] == 'AMD':
        model_options = amd_models
        if selected_line in ['Inspiron', 'Vostro']: model_weights = [0.55, 0.44, 0.01] # Heavy R3/R5
        else: model_weights = [0.1, 0.6, 0.3] # R3/R5/R7 Mix
        row['CPU_Model'] = random.choices(model_options, weights=model_weights, k=1)[0]
        row['CPU_Generation'] = random.choices(['5000', '6000', '7000'], weights=[0.15, 0.35, 0.5], k=1)[0] # Favor 6k/7k


    # RAM (8 or 16 GB only)
    ram_weights = {8: 0.65, 16: 0.35} # Heavily favor 8GB
    if selected_line in ['Latitude', 'G Series', 'XPS']: ram_weights = {8: 0.2, 16: 0.8} # Favor 16GB for these
    row['RAM_GB'] = random.choices(list(ram_options), weights=[ram_weights[r] for r in ram_options], k=1)[0]
    # RAM_Type column removed

    # Storage (Focus on 256/512GB)
    storage_weights = {256: 0.6, 512: 0.35, 1000: 0.05} # Heavy 256/512
    if selected_line in ['Latitude','G Series', 'XPS']: storage_weights = {256: 0.1, 512: 0.6, 1000: 0.3} # Favor 512/1TB
    row['Storage_GB'] = random.choices(list(storage_options), weights=[storage_weights[s] for s in storage_options], k=1)[0]


    # GPU Type, Name, VRAM (Very few dedicated)
    gpu_type_weights = {'Integrated': 0.92, 'Dedicated': 0.08} # Even more Integrated
    if selected_line in ['G Series', 'XPS']: gpu_type_weights = {'Integrated': 0.1, 'Dedicated': 0.9} # Still allow dedicated for G/XPS

    row['GPU_Type'] = random.choices(list(gpu_type_weights.keys()), weights=list(gpu_type_weights.values()), k=1)[0]

    if row['GPU_Type'] == 'Integrated':
        if row['CPU_Vendor'] == 'Intel':
            row['GPU_Name'] = 'Intel Iris Xe' if row['CPU_Model'] in ['Core i5', 'Core i7', 'Core Ultra 5'] else 'Intel UHD Graphics'
        elif row['CPU_Vendor'] == 'AMD':
            row['GPU_Name'] = 'AMD Radeon Graphics'
        row['GPU_VRAM_GB'] = 0
    else: # Dedicated (Very limited options)
        gpu_weights = {
             'NVIDIA MX550': 0.5,
             'AMD Radeon RX 6500M': 0.3,
             'NVIDIA RTX 3050': 0.2,
        }
        # Select name directly
        row['GPU_Name'] = random.choices(list(gpu_weights.keys()), weights=list(gpu_weights.values()), k=1)[0]
        # Get VRAM based on selected name
        possible_vram = gpu_vram_options.get(row['GPU_Name'], [0])
        row['GPU_VRAM_GB'] = random.choice(possible_vram) if possible_vram else 0


    # Screen Specs (No Size/Resolution)
    panel_weights = {'Standard': 0.65, 'IPS': 0.35} # OLED removed from options
    if selected_line in ['XPS', 'G Series', 'Latitude']: panel_weights = {'Standard': 0.1, 'IPS': 0.9} # Force IPS for higher lines
    row['Screen_Panel_Type'] = random.choices(list(screen_panels), weights=[panel_weights[p] for p in screen_panels], k=1)[0]

    refresh_weights = {60: 0.9, 90: 0.07, 120: 0.03} # Very heavy 60Hz
    if selected_line in ['G Series', 'XPS']: refresh_weights = {60: 0.1, 90: 0.2, 120: 0.7} # Allow higher for G/XPS
    row['Screen_Refresh_Rate_Hz'] = random.choices(list(screen_refresh_rates), weights=[refresh_weights[r] for r in screen_refresh_rates], k=1)[0]

    touch_prob = 0.05 # Very low touch prob
    if selected_line in ['Latitude', 'XPS']: touch_prob = 0.15
    row['Screen_Touch'] = 'Yes' if random.random() < touch_prob else 'No'


    # Battery
    batt_weights = {41: 0.5, 54: 0.45, 60: 0.05} # Focus on smallest
    if selected_line in ['XPS', 'Latitude', 'G Series']: batt_weights = {41:0.2, 54: 0.5, 60: 0.3}
    row['Battery_Whr'] = random.choices(list(battery_options), weights=[batt_weights[b] for b in battery_options], k=1)[0]

    # Calculate Price using adjusted function and strict 50M cap
    row['Price_VND'] = calculate_price(row)

    # Append row
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# --- Final Column Ordering (Removed Columns) ---
final_column_order = [
    'Model_Name',
    'Product_Line',
    'Segment',
    'CPU_Vendor',
    'CPU_Model',
    'CPU_Generation',
    'GPU_Type',
    # 'GPU_Vendor', # REMOVED
    'GPU_Name',
    'GPU_VRAM_GB',
    'RAM_GB',
    # 'RAM_Type', # REMOVED
    'Storage_GB',
    # 'Storage_Type', # REMOVED
    # 'Screen_Size_Inches', # REMOVED
    # 'Screen_Resolution', # REMOVED
    'Screen_Panel_Type',
    'Screen_Refresh_Rate_Hz',
    'Screen_Touch',
    'Battery_Whr',
    'Price_VND'
]

# Reorder DataFrame
df = df[final_column_order]

# Save to CSV
output_filename = 'dataset18.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"Generated dataset with {len(df)} rows and saved to {output_filename}")
print(f"\nStrict Maximum Price Cap Applied: 50,000,000 VND")
print("\nPrice Distribution Summary:")
print(df['Price_VND'].describe())

# Verify Price Range Focus
price_10_25M = df[(df['Price_VND'] >= 10_000_000) & (df['Price_VND'] <= 25_000_000)].shape[0]
price_over_30M = df[df['Price_VND'] > 30_000_000].shape[0]
price_over_40M = df[df['Price_VND'] > 40_000_000].shape[0]
price_at_50M = df[df['Price_VND'] == 50_000_000].shape[0]

print(f"\nLaptops in 10-25M VND range: {price_10_25M} ({price_10_25M / num_rows * 100:.1f}%)")
print(f"Laptops over 30M VND: {price_over_30M} ({price_over_30M / num_rows * 100:.1f}%)")
print(f"Laptops over 40M VND: {price_over_40M} ({price_over_40M / num_rows * 100:.1f}%)")
print(f"Laptops exactly at 50M VND cap: {price_at_50M}")


# Verify Product Line distribution
print("\nProduct Line Distribution:")
print(df['Product_Line'].value_counts(normalize=True))

print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())