# OpenMM Constant-Voltage MD Optimization

本專案包含基於 **OpenMM + SAPT-FF** 的定電壓分子動力學（Fixed-Voltage Molecular Dynamics / Supercapacitor Electrode Simulation）原始參考實作與高性能優化實作，並在完全不改變物理結果與模擬軌跡（達 IEEE-754 雙精度 FP64 機器精度一致）的前提下，大幅加速初始化、Poisson 電荷自洽求解器與 MD 積分速度。

---

## 目錄結構

```text
test_optimization/
├── OpenMM_ConstantV_original/    # 原始參考實作（完全保持未修改，作為對照基準）
│   ├── ffdir/                    # SAPT-FF 與石墨烯/奈米碳管力場定義 XML
│   ├── lib/
│   │   ├── MM_classes.py
│   │   ├── Fixed_Voltage_routines.py
│   │   ├── electrode_sapt_exclusions.py
│   │   └── add_customnonbond_xml.py
│   ├── nvt_0V_15ns.pdb           # 電極-離子液體電解質初始構型 (17,152 粒子)
│   ├── sapt_exclusions.py
│   ├── run_openMM.py
│   └── run.sh
├── OpenMM_ConstantV_optimized/   # 優化版實作（與原始模擬結果 100% 數值對齊）
│   ├── ffdir/
│   ├── lib/
│   ├── nvt_0V_15ns.pdb
│   ├── sapt_exclusions.py
│   ├── run_openMM.py
│   ├── run.sh
│   └── verify_parity.py          # 7 大項雙精度一致性與效能基準自動化驗證套件
├── pixi.toml                     # Pixi 環境與快捷任務配置 (OpenMM 8.5+, CUDA, NumPy)
└── .gitignore
```

---

## 核心優化項目摘要

1. **`CustomNonbondedForce` 交互作用群組化 (`addInteractionGroup`)**：
   - 自動篩選非零參數的真實電解質原子與真實電極原子建立 `InteractionGroup`（並支援混合水模型 `HOH`、巴克球 `Buckyball_Virtual`、奈米碳管 `Nanotube_Virtual` 與純真空電極系統）。
   - 徹底排除 `6,252` 個全零參數粒子（虛擬電極 `Cgtc`、Drude 振子 `Sh`、虛擬氫 `Hgt`，佔系統總粒子數 **36.5%**）的 GPU Neighbor List 配對與所有電極內部非鍵計算，免除對電極呼叫 253 萬次 `customNonbondedForce.addExclusion`（排除項數量由 **`2,634,890` 降至 `102,110`，減少 96.1%**）。
2. **`NonbondedForce` 冗餘電極 Exceptions 精簡**：
   - 保留虛擬電極同平面內的 `nbondedForce.addException` 以確保總位能純量 `getPotentialEnergy()` 與原始模擬完全一致（連常數自能偏移都沒有），並自動略過真實電極層（`electrode_extra_exclusions`）中 `charge == 0` 且 `epsilon == 0` 的 190.8 萬筆無效 Exceptions（Exceptions 數量由 **`2,634,890` 降至 `726,910`，減少 72.4%**），大幅降低 `updateParametersInContext` 與每步 PME 排除力 Kernel 的開銷。
3. **快取靜態電解質電荷與 NumPy 向量化 `Poisson_solver_fixed_voltage`**：
   - 於 `initialize_electrolyte()` 快取電解質靜態電荷陣列 `electrolyte_charges_np`，將每次呼叫 `compute_Electrode_charge_analytic` 的 **27,900 次 SWIG C++ `getParticleParameters` 查詢**化簡為單行 NumPy 向量運算（加速 **824 倍**）。
   - SCF 迴圈內的 `getState` 指定僅評估 `NonbondedForce` 的 Force Group (`groups={self.nbondedForce.getForceGroup()}`)，跳過昂貴的 `CustomNonbondedForce` 與鍵結力場計算，並全面啟用 `asNumpy=True`。
   - 以 NumPy 向量化陰極、陽極與額外導體的電荷更新與正規化縮放，消除每次迭代重複的 `setParticleParameters` 與迴圈後冗餘的寫入；透過 `@property` 將 `atom_MM.charge` 直接綁定至 `electrode_charges_np`，確保外部屬性讀寫與內部向量陣列零誤差同步。
4. **向量化 `MC_Barostat_step` 與初始化 `reinitialize(preserveState=True)`**：
   - 預先快取陽極原子索引與電解質分子參考原子索引，以 NumPy 向量平移取代雙層 Python 迴圈。
   - 在 `generate_exclusions_TFSI` 與 `sapt_exclusions.py` 中移除 263 萬筆無效的 `flagexceptions` 掃描，並以整數 tuple 集合與 `reinitialize(preserveState=True)` 加速啟動。

---

## 數值一致性與效能基準（NVIDIA GeForce RTX 4060 GPU, OpenMM 8.5.1）

### 1. 雙精度（CUDA FP64 / Reference FP64）模擬一致性驗證 (`verify_parity.py`)

| 驗證測試項目 | 原始版 vs. 優化版 最大差異 | 狀態 |
| :--- | :--- | :---: |
| **1. 全部 10 個 Force Groups 位能與每原子受力** | `NonbondedForce`: $\Delta E = 2.33 \times 10^{-10}\text{ kJ/mol}$, $\max\|\Delta F\| = 1.86 \times 10^{-8}$<br>`CustomNonbondedForce`: $\Delta E = 1.09 \times 10^{-11}\text{ kJ/mol}$, $\max\|\Delta F\| = 4.42 \times 10^{-9}$<br>其餘 8 個力場群組：$\Delta E = 0.0$, $\max\|\Delta F\| = 0.0$ | ✅ 通過 |
| **2. 多電壓 (`0.0V`, `1.0V`, `-1.5V`) Poisson 收斂電荷** | $\max\|\Delta q_{\text{cathode}}\| \le 9.08 \times 10^{-9}\ e$, $\max\|\Delta q_{\text{anode}}\| \le 1.34 \times 10^{-8}\ e$, $\|\Delta Q_{\text{analytic}}\| \le 1.65 \times 10^{-13}\ e$ | ✅ 通過 |
| **3. 多步 Constant-V MD 動態軌跡 (`250` 步)** | $\Delta PE = 6.77 \times 10^{-7}\text{ kJ/mol}$, $\max\|\Delta \mathbf{r}\| = 1.63 \times 10^{-9}\text{ nm}$ | ✅ 通過 |
| **4. `MC_Barostat_step` 蒙地卡羅平衡** | 接受率完全相同，$\Delta L_{\text{cell}} = 0.0\text{ nm}$, $\max\|\Delta \mathbf{r}\| = 3.69 \times 10^{-12}\text{ nm}$ | ✅ 通過 |
| **5. 純真空電極系統 (`noions`) `CustomNonbondedForce`** | 原始版 `0.0 kJ/mol` vs. 優化版 `0.0 kJ/mol` ($\Delta E = 0.0$) | ✅ 通過 |
| **6. `Buckyball_Virtual` 與 `Nanotube_Virtual` 導體電荷** | $\max\|\Delta q_{\text{cond}}\| \le 1.50 \times 10^{-9}\ e$，`atom.charge` 同步誤差 `0.0` | ✅ 通過 |
| **7. 混合水模型 (`HOH`) 與獨立 `sapt_generate_exclusions`** | 全力場群組 $\Delta E < 10^{-7}\text{ kJ/mol}$；獨立排除項 $\Delta PE = 0.0$ | ✅ 通過 |

### 2. 效能加速對比（RTX 4060 GPU, CUDA Mixed Precision）

| 評估項目 | 原始版 (`OpenMM_ConstantV_original`) | 優化版 (`OpenMM_ConstantV_optimized`) | 變化 / 加速倍率 |
| :--- | :---: | :---: | :---: |
| `CustomNonbondedForce` Exclusions 數 | `2,634,890` (0 groups) | **`102,110` (2 groups)** | **-96.1%** |
| `NonbondedForce` Exceptions 數 | `2,634,890` | **`726,910`** | **-72.4%** |
| `generate_exclusions` 耗時 | `9.80 s` | **`2.06 s`** | **4.77x 加速** |
| 系統總初始化耗時 | `11.58 s` | **`3.76 s`** | **3.08x 加速** |
| `Poisson_solver_fixed_voltage(Niterations=4)` | `141.61 ms` / call | **`24.70 ms` / call** | **5.73x 加速** |
| `simmd.step(200)` (每 200 fs MD 積分) | `347.11 ms` | **`288.47 ms`** | **1.20x 加速** |

---

## 快速開始（使用 Pixi）

本專案使用 [Pixi](https://pixi.sh) 管理相依套件，無須 `sudo` 權限即可執行：

```bash
# 1. 安裝/同步環境（若已安裝則可加上 --frozen）
pixi install

# 2. 執行原始版 vs. 優化版雙精度對齊與效能基準驗證
pixi run --frozen verify

# 3. 執行優化版定電壓 MD 模擬
pixi run --frozen run-optimized

# 4. 執行原始版定電壓 MD 模擬
pixi run --frozen run-original
```
