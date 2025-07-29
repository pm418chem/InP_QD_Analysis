import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob

# Expected input is an Excel file multiple sheets. Each sheet has a different set of data in.
# The first column is wavelength in nm, the second column is energy in eV, and subsequent columns are absorption data.

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

excel_files = glob.glob("*.xlsx")

for excel_path in excel_files:
    base = os.path.splitext(os.path.basename(excel_path))[0]
    results = []

    #make a directory for the fit plots
    os.makedirs(f'fit_plots/{base}', exist_ok=True)

    # Load all sheets
    sheets = pd.read_excel(excel_path, sheet_name=None)
    
    for sheet_name, df in sheets.items():
        if df.iloc[:, 0].isnull().all():
            print(f"First column in {base} | {sheet_name} is empty, dropping column and retrying.")
            df = df.drop(df.columns[0], axis=1)
            # Check if new first column is also empty
            if df.shape[1] == 0 or df.iloc[:, 0].isnull().all():
                print(f"Second column in {base} | {sheet_name} is also empty, skipping this Excel file.")
                break  # Move to next Excel file
        
        x = df.iloc[:, 1].values  # Second column is energy (x-axis)

        for col in df.columns[2:]:  # Columns 3 onward are absorption (y-axis)
            y = df[col].values
            # Skip if y is all NaN or empty
            if np.all(pd.isna(y)) or len(y) == 0:
                print(f"No data in column '{col}', skipping.")
                continue

            # Optionally, skip if there are fewer than 3 non-NaN values
            if np.count_nonzero(~np.isnan(y)) < 3:
                print(f"Not enough data in column '{col}', skipping.")
                continue
            
            # Only interested in fitting the 1.7 - 3 eV region
            while True:
                # Mask for 1.7 - 3 eV region
                x_zoom_mask = (x >= 1.7) & (x <= 3)
                x_zoom = x[x_zoom_mask]
                y_zoom = y[x_zoom_mask]
                
                sheet_label = "" if sheet_name.lower() == "sheet1" else sheet_name

                # Asking the user to click near the shoulder
                fig, ax = plt.subplots()
                ax.plot(x, y, 'b.', alpha=0.3, label='All Data')
                ax.plot(x_zoom, y_zoom, 'ro', label='Zoom Region')
                ax.set_xlim(1.7, 3)
                if len(y_zoom) > 0:
                    ax.set_ylim(y_zoom.min(), y_zoom.max())
                ax.set_title(f"{base} | {sheet_label} | {col}\nClick near the shoulder")
                ax.set_xlabel('Energy (eV)')
                ax.set_ylabel('Absorption')
                plt.legend()
                plt.tight_layout()
                print(f"Please click near the shoulder for {base} | {sheet_name} | {col} and close the plot window.")
                clicked = plt.ginput(1, timeout=-1)
                plt.close(fig)

                if not clicked:
                    print(f"No click detected for {col}, skipping.")
                    break

                x_click = clicked[0][0]

                window = 0.1
                mask = (x > x_click - window) & (x < x_click + window)
                if not np.any(mask):
                    print(f"No data near clicked point for {col}, skipping.")
                    break
                local_idx = np.argmax(y[mask])
                shoulder_x = x[mask][local_idx]

                fit_mask = x < shoulder_x
                x_fit = x[fit_mask]
                y_fit = y[fit_mask]
                if len(x_fit) < 3:
                    print(f"Not enough data to fit for {col}, skipping.")
                    break
                
                # Fit a Gaussian to the data
                # Remove offset for fitting
                offset = y[0]
                y_fit_offset = y_fit - offset
                p0 = [y_fit_offset.max(), x_fit[np.argmax(y_fit_offset)], 0.1]
                try:
                    popt, pcov = curve_fit(gaussian, x_fit, y_fit_offset, p0=p0)
                except RuntimeError:
                    print(f"Fit failed for {col}")
                    break

                # Calculate fitted values
                y_fit_pred = gaussian(x_fit, *popt) + offset

                # R² calculation
                ss_res = np.sum((y_fit - y_fit_pred) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Show the fit
                x_zoom_mask = (x >= 1.7) & (x <= 3)
                x_zoom = x[x_zoom_mask]
                y_zoom = y[x_zoom_mask]

                plt.figure()
                plt.plot(x, y, 'b.', alpha=0.3, label='All Data')
                plt.plot(x_zoom, y_zoom, 'ro', label='Zoom Region')
                plt.axvline(popt[1], color='gray', linestyle='--', label='Peak Centre')
                plt.plot(x, gaussian(x, *popt), 'g-', label='Gaussian fit')
                plt.xlabel('Energy (eV)')
                plt.ylabel('Absorption (a.u.)')
                plt.title(f'Gaussian Fit ({base} | {sheet_label} | {col})')
                plt.xlim(1.7, 3)
                if len(y_zoom) > 0:
                    plt.ylim(y_zoom.min(), y_zoom.max())
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()

                # Ask user if they accept the fit
                accept = input("Do you want to accept this fit? (y/n/b for bad fit): ").strip().lower()
                if accept == 'y':
                    # Save the plot
                    plt.figure()
                    plt.plot(x, y, 'b.', alpha=0.3, label='All Data')
                    #plt.plot(x_zoom, y_zoom, 'ro', label='Zoom Region')
                    plt.axvline(popt[1], color='gray', linestyle='--', label='Peak Centre')
                    plt.plot(x, gaussian(x, *popt) + offset, 'g-', label='Gaussian fit')
                    plt.xlabel('Energy (eV)')
                    plt.ylabel('Absorption')
                    plt.title(f'Gaussian Fit ({base} {sheet_label} {col})')
                    plt.xlim(1.7, 3)
                    if len(y_zoom) > 0:
                        plt.ylim(y_zoom.min(), y_zoom.max())
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'fit_plots/{base}/{base}_{sheet_label}_{col}_fit.png')
                    plt.close()

                    results.append({
                        'file': base,
                        'sheet': sheet_label,
                        'dataset': col,
                        'amplitude': popt[0],
                        'center': popt[1],
                        'sigma': popt[2],
                        'offset': offset,
                        'shoulder_x': shoulder_x,
                        'r_squared': r_squared,
                        'comment': ''
                    })
                    break
                elif accept == 'b':
                    print("Marked as bad fit, moving on.")
                    # Save the plot
                    plt.figure()
                    plt.plot(x, y, 'b.', alpha=0.3, label='All Data')
                    #plt.plot(x_zoom, y_zoom, 'ro', label='Zoom Region')
                    plt.axvline(popt[1], color='gray', linestyle='--', label='Peak Centre')
                    plt.plot(x, gaussian(x, *popt) + offset, 'g-', label='Gaussian fit')
                    plt.xlabel('Energy (eV)')
                    plt.ylabel('Absorption')
                    plt.title(f'Gaussian Fit ({base} {sheet_label} {col})')
                    plt.xlim(1.7, 3)
                    if len(y_zoom) > 0:
                        plt.ylim(y_zoom.min(), y_zoom.max())
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'fit_plots/{base}/{base}_{sheet_label}_{col}_fit.png')
                    plt.close()
                    
                    results.append({
                        'file': base,
                        'sheet': sheet_label,
                        'dataset': col,
                        'amplitude': popt[0],
                        'center': popt[1],
                        'sigma': popt[2],
                        'offset': offset,
                        'shoulder_x': shoulder_x,
                        'r_squared': r_squared,
                        'comment': 'bad fit'
                    })
                    break
                else:
                    print("Let's try again!")
    # Save results for this Excel file
    if results:
        results_df = pd.DataFrame(results)
        csv_filename = f"{base}_gaussian_fits.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")
