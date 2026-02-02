# Licensing and Pricing Analysis

## Document Recognition System - Software Components

This document provides a comprehensive analysis of licensing terms, commercial use implications, and pricing for all software components used in this project.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Detailed Licensing Information](#detailed-licensing-information)
3. [Commercial Use Analysis](#commercial-use-analysis)
4. [Cloud OCR Alternatives Pricing](#cloud-ocr-alternatives-pricing)
5. [Cost Comparison](#cost-comparison)
6. [Recommendations](#recommendations)

---

## Executive Summary

### Software Components Used

| Component | Version | License | Commercial Use | Cost |
|-----------|---------|---------|----------------|------|
| **Surya OCR** | ≥0.4.0 | GPL-3.0 (code) + Modified Open Rail-M (weights) | Restricted | Free < $2M revenue |
| **OpenAI CLIP** | clip-vit-base-patch32 | MIT | ✅ Permitted | Free |
| **FastAPI** | ≥0.104.0 | MIT | ✅ Permitted | Free |
| **Uvicorn** | ≥0.24.0 | BSD-3-Clause | ✅ Permitted | Free |
| **OpenCV** | ≥4.8.0 | Apache 2.0 | ✅ Permitted | Free |
| **PyTorch** | ≥2.0.0 | BSD-3-Clause | ✅ Permitted | Free |
| **Transformers** | ≥4.36.0 | Apache 2.0 | ✅ Permitted | Free |
| **SciPy** | ≥1.10.0 | BSD-3-Clause | ✅ Permitted | Free |
| **NumPy** | ≥1.24.0 | BSD-3-Clause | ✅ Permitted | Free |
| **Pillow** | ≥10.0.0 | HPND | ✅ Permitted | Free |
| **Jinja2** | ≥3.1.0 | BSD-3-Clause | ✅ Permitted | Free |
| **python-multipart** | ≥0.0.6 | Apache 2.0 | ✅ Permitted | Free |
| **aiofiles** | ≥23.0.0 | Apache 2.0 | ✅ Permitted | Free |
| **requests** | ≥2.31.0 | Apache 2.0 | ✅ Permitted | Free |

### Key Finding

**⚠️ Primary Licensing Concern: Surya OCR**

Organizations with >$2M revenue or funding require a commercial license from Datalab.

---

## Detailed Licensing Information

### 1. Surya OCR ⚠️

**Purpose**: Text detection and recognition (OCR engine)

**Repository**: https://github.com/datalab-to/surya

**License**:
- **Code**: GPL-3.0 (GNU General Public License v3.0)
- **Model Weights**: Modified AI Pubs Open Rail-M

**Commercial Terms** (Source: Datalab):

> "Our model weights use a modified AI Pubs Open Rail-M license (free for research, personal use, and startups under $2M funding/revenue) and our code is GPL."

**Free Use Permitted**:
- ✅ Research and academic use
- ✅ Personal projects
- ✅ Startups with < $2M gross revenue (trailing 12 months)
- ✅ Startups with < $2M investor capital raised

**Commercial License Required**:
- ❌ Organizations with > $2M gross revenue
- ❌ Organizations with > $2M investor funding
- ❌ Use that directly generates or saves money commercially

**GPL-3.0 Implications**:
- Modifications to Surya code must be released under GPL
- Distributing software containing Surya may require entire application to be GPL
- Using Surya as an internal service (not distributed) may avoid GPL requirements

**Commercial Licensing**: https://www.datalab.to/pricing

---

### 2. OpenAI CLIP

**Purpose**: Document image classification (zero-shot)

**Model**: `openai/clip-vit-base-patch32`

**Repository**: https://github.com/openai/CLIP

**License**: MIT License

```
MIT License
Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

**Commercial Use**: ✅ Fully permitted without restrictions

**Cost**: Free

---

### 3. FastAPI

**Purpose**: Web framework for REST API

**Repository**: https://github.com/fastapi/fastapi

**License**: MIT License

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

**Documentation**: https://fastapi.tiangolo.com/

---

### 4. Uvicorn

**Purpose**: ASGI server for FastAPI

**Repository**: https://github.com/encode/uvicorn

**License**: BSD-3-Clause

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

### 5. OpenCV (opencv-python)

**Purpose**: Image processing, deskewing, rotation

**Repository**: https://github.com/opencv/opencv

**License**: Apache License 2.0

> "OpenCV is open source and released under the Apache 2 License. It is free for commercial use."
> — opencv.org

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

**Documentation**: https://docs.opencv.org/4.x/

---

### 6. PyTorch

**Purpose**: Deep learning framework (required by Surya and CLIP)

**Repository**: https://github.com/pytorch/pytorch

**License**: BSD-3-Clause

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

**Documentation**: https://pytorch.org/docs/

---

### 7. Transformers (Hugging Face)

**Purpose**: Model loading and inference for CLIP

**Repository**: https://github.com/huggingface/transformers

**License**: Apache License 2.0

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

**Note**: Individual models on Hugging Face may have different licenses. CLIP uses MIT.

**Documentation**: https://huggingface.co/docs/transformers/

---

### 8. SciPy

**Purpose**: Image rotation for deskew algorithm (ndimage.rotate)

**Repository**: https://github.com/scipy/scipy

**License**: BSD-3-Clause

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

**Documentation**: https://docs.scipy.org/doc/scipy/

---

### 9. NumPy

**Purpose**: Numerical operations, array processing

**Repository**: https://github.com/numpy/numpy

**License**: BSD-3-Clause

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

### 10. Pillow

**Purpose**: Image loading and manipulation

**Repository**: https://github.com/python-pillow/Pillow

**License**: HPND (Historical Permission Notice and Disclaimer) - permissive

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

### 11. Jinja2

**Purpose**: HTML templating

**Repository**: https://github.com/pallets/jinja

**License**: BSD-3-Clause

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

### 12. python-multipart

**Purpose**: File upload handling

**License**: Apache License 2.0

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

### 13. aiofiles

**Purpose**: Async file operations

**License**: Apache License 2.0

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

### 14. requests

**Purpose**: HTTP client library

**License**: Apache License 2.0

**Commercial Use**: ✅ Fully permitted

**Cost**: Free

---

## Commercial Use Analysis

### License Categories

**Permissive Licenses (No Restrictions)**:
- MIT: CLIP, FastAPI
- Apache 2.0: OpenCV, Transformers, python-multipart, aiofiles, requests
- BSD-3-Clause: PyTorch, SciPy, NumPy, Uvicorn, Jinja2
- HPND: Pillow

**Copyleft/Restricted Licenses**:
- GPL-3.0: Surya OCR (code)
- Modified Open Rail-M: Surya OCR (weights) - revenue threshold

### Surya OCR Commercial Licensing Details

**Revenue/Funding Thresholds**:

| Organization Type | Threshold | License Required |
|-------------------|-----------|------------------|
| Startups | < $2M revenue OR < $2M funding | Free |
| Small Business | $2M - $5M revenue | Commercial License |
| Enterprise | > $5M revenue | Commercial License |

**Contact for Pricing**: https://www.datalab.to/pricing

**Alternative**: Datalab offers a hosted API service with pay-per-use pricing.

---

## Cloud OCR Alternatives Pricing

For comparison, here are pricing for major cloud OCR services:

### Google Cloud Vision API

**Pricing** (per 1,000 units/pages):

| Volume (monthly) | Text Detection (OCR) |
|------------------|---------------------|
| First 1,000 | Free |
| 1,001 - 5,000,000 | $1.50 |
| 5,000,001+ | $0.60 |

**Documentation**: https://cloud.google.com/vision/pricing

---

### Amazon Textract

**Pricing** (US West Oregon region):

| Feature | First 1M pages | After 1M pages |
|---------|----------------|----------------|
| Detect Text (OCR) | $0.0015/page | $0.0006/page |
| Forms + Tables | $0.065/page | $0.050/page |
| Queries | $0.015/page | $0.010/page |

**Free Tier**: 1,000 pages/month for first 3 months

**Documentation**: https://aws.amazon.com/textract/pricing/

---

### Microsoft Azure Document Intelligence

**Pricing**:

| Feature | Price per 1,000 pages |
|---------|----------------------|
| Read (OCR) | $1.50 |
| Layout | $10.00 |
| Prebuilt Models | $10.00 |
| Custom Extraction | $30.00 |

**Free Tier**: 500 pages/month

**Documentation**: https://azure.microsoft.com/en-us/pricing/details/document-intelligence/

---

### ABBYY FineReader

**Desktop Software Pricing** (subscription):

| Edition | Annual | 3-Year |
|---------|--------|--------|
| Standard | £84/year | £227 |
| Corporate | £139/year | £375 |
| Mac | £59/year | £227 |

**Server/SDK**: Contact for enterprise pricing ($3,000 - $15,000+ per server)

**Documentation**: https://pdf.abbyy.com/pricing/

---

### Tesseract OCR (Open Source Alternative)

**License**: Apache 2.0

**Cost**: Free

**Commercial Use**: ✅ Fully permitted

**Limitations**:
- Lower accuracy than modern transformer-based OCR
- Requires more preprocessing
- No built-in document classification

**Repository**: https://github.com/tesseract-ocr/tesseract

---

## Cost Comparison

### Scenario: 10,000 pages/month

| Solution | Monthly Cost | Annual Cost |
|----------|-------------|-------------|
| **This Project (Local)** | $0* | $0* |
| Google Cloud Vision | $13.50 | $162 |
| Amazon Textract | $15.00 | $180 |
| Azure Document Intelligence | $15.00 | $180 |
| ABBYY FineReader Corporate | £11.58 (~$15) | £139 (~$175) |

*Free if organization is under $2M revenue threshold. Otherwise, Surya commercial license required.

### Scenario: 100,000 pages/month

| Solution | Monthly Cost | Annual Cost |
|----------|-------------|-------------|
| **This Project (Local)** | $0* | $0* |
| Google Cloud Vision | $148.50 | $1,782 |
| Amazon Textract | $150.00 | $1,800 |
| Azure Document Intelligence | $150.00 | $1,800 |

### Break-Even Analysis

For high-volume processing, local deployment saves significant costs:
- **100K pages/month**: ~$1,800/year savings vs cloud
- **1M pages/month**: ~$18,000/year savings vs cloud

**Additional Local Deployment Costs**:
- Server hardware (one-time): $2,000 - $10,000
- GPU (recommended): $500 - $2,000
- Electricity and maintenance: ~$50-100/month

---

## Recommendations

### For Startups (< $2M revenue/funding)

✅ **Use this project as-is**
- All components are free for your use case
- No licensing concerns
- Significant cost savings vs cloud services

### For Growing Companies ($2M - $5M revenue)

⚠️ **Contact Datalab for Surya commercial license**
- Required for legal compliance
- Alternative: Switch to Tesseract (lower accuracy) or cloud services
- Cost-benefit analysis needed based on volume

### For Enterprise (> $5M revenue)

**Options**:

1. **Surya Commercial License**
   - Contact Datalab for enterprise pricing
   - Best accuracy, full local control

2. **Cloud Services**
   - Google Cloud Vision, AWS Textract, or Azure
   - Pay-per-use, no licensing concerns
   - Higher per-page cost at scale

3. **Alternative Open Source**
   - Replace Surya with Tesseract (Apache 2.0)
   - Lower accuracy but fully permissive
   - May require additional preprocessing

### GPL Compliance Notes

If distributing software containing Surya:
1. Entire application may need to be GPL-licensed
2. Source code must be made available
3. Consider using Surya as a separate microservice to isolate GPL requirements

---

## References

- Surya OCR: https://github.com/datalab-to/surya
- Datalab Pricing: https://www.datalab.to/pricing
- OpenAI CLIP: https://github.com/openai/CLIP
- FastAPI: https://fastapi.tiangolo.com/
- OpenCV License: https://opencv.org/license/
- PyTorch: https://pytorch.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Google Cloud Vision Pricing: https://cloud.google.com/vision/pricing
- AWS Textract Pricing: https://aws.amazon.com/textract/pricing/
- Azure Document Intelligence: https://azure.microsoft.com/en-us/pricing/details/document-intelligence/
- ABBYY Pricing: https://pdf.abbyy.com/pricing/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract

---

*Document generated: January 2026*
*Prices and licensing terms subject to change. Verify with official sources before making decisions.*
