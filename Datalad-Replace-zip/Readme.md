# 🧠 DataLad ZIP Preparation Workflow  
### Folder-Driven, Resumable, Manual-Control First

This repository provides a **safe and resumable workflow** for preparing large ZIP archives in a **DataLad / git-annex dataset** before manual inspection, extraction, and later committing.

The core idea is simple:

> 📁 **Folders are truth. ZIPs are not and should be avoided/not used in DataLad datasets**

---

## 🚦 Core Behavior (at a Glance)

✅ Iterate **one ZIP at a time**  
✅ Resume safely after interruption  
✅ Skip anything already completed  
❌ No unzip  
❌ No save / push / drop  
🧑‍🔬 Manual inspection encouraged  

---

## 🔑 Key Rule (Very Important)

### 🟢 Continue **only if the folder is missing**

| Condition | Action |
|---------|--------|
| 📁 Folder exists | ⏭️ Skip completely |
| 📁 Folder missing | ▶️ Process ZIP |

➡️ ZIP state (present, unlocked, partial, failed) is **ignored**

This guarantees:
- no accidental overwrites
- no false positives from partial work
- safe reruns across commits and machines

---

## 🔄 What the Script Does

For each `*.zip` file:

1️⃣ Derives the target folder name  
2️⃣ Checks whether that folder already exists  
3️⃣ If missing:
- 📥 `datalad get`
- 🔓 `datalad unlock`
4️⃣ Moves on to the next ZIP  

🛑 **After every N ZIPs (default: 5):**
- pauses
- waits for user input
- allows manual inspection

---

## 🧑‍🔬 What the Script Intentionally Does *NOT* Do

❌ No unzip (ZIPs may be huge / slow / fragile)  
❌ No `datalad save`  
❌ No `datalad push`  
❌ No `datalad drop`  

➡️ All destructive or irreversible actions are **user-controlled**

---

## 📦 Why Folder-Driven Logic?

ZIP-based logic is unreliable because ZIPs can be:

⚠️ partially downloaded  
⚠️ previously unlocked  
⚠️ present but never validated  
⚠️ corrupted but still large  

Folders, however, mean:

✅ data extracted  
✅ structure verified  
✅ work completed  

If a folder exists, the script assumes:
> “This dataset entry is done.”

---

## ⏸️ Batch Processing & Pausing

- Default batch size: **5**
- After each batch:
  - ⏸️ script pauses
  - 👀 user inspects ZIPs
  - 🧪 optional manual unzip & validation
  - ▶️ user resumes explicitly

Perfect for:
- slow networks
- large archives
- long annex transfers

---

## 🔁 Safe to Re-Run (Any Time)

You can re-run the script:

- after crashes
- after switching commits
- after partial downloads
- after manual cleanup

Already existing folders are **never touched again**.

---

## 🧭 Typical Workflow

🔹 Run script  
🔹 Wait for pause  
🔹 Inspect ZIPs  
🔹 Unzip manually  
🔹 Validate folder size / content  
🔹 Later: run a **separate** save/push/drop script  

---

## 🛡️ Safety Guarantees

✔️ No automatic extraction  
✔️ No automatic commits  
✔️ No data deletion  
✔️ No forced retries  
✔️ Errors logged, loop continues  

---

## 🎯 Intended Use Case

- Large MRI / neuroimaging datasets  
- git-annex backed storage  
- Long-running or unstable transfers  
- Human-in-the-loop QA before commits  

---

## 📜 License

Use freely.  
No warranty.  
You break it, you keep both pieces 😉
