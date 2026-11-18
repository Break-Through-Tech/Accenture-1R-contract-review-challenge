# Demo contracts

Sample inputs for the demo page's "Try the sample contracts" button (`python -m clausetriage.ui.app`).
They are for showing the pipeline only; none of them is used for training or evaluation.

| File | Source |
|---|---|
| `cuad_validation_fuelcell_development_agreement.pdf` | Text of the CUAD contract `FuelcellEnergyInc_20191106_8-K_EX-10.1_11868007_EX-10.1_Development Agreement`, rendered to a plain text PDF. It is in the validation split (taken from `train_separate_questions.json`, seed 42), not `test.json`. CUAD is licensed CC BY 4.0 (The Atticus Project). |
| `nasa_kbr_agreement.pdf` | Signed NASA Space Act Agreement with KBR (SAA-CA-24-40666), published by NASA at https://nasa.gov/wp-content/uploads/static/saa/domestic/40666_40666_KBR_Agreement_Fully_Signed.pdf |
| `nasa_sierra_space_umbrella_agreement.pdf` | Signed NASA Space Act umbrella agreement with Sierra Space, published by NASA at https://www.nasa.gov/saa/domestic/42865_Sierra_Space_SAA_Umbrella_PAM_42865-Executed.pdf |

The NASA agreements are government agreements rather than the commercial contracts CUAD was built
from, so the model detects fewer clauses in them and they score low.
