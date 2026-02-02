// Application state
const state = {
    images: [],
    selectedImageId: null,
    ocrResult: null,
    imageScale: 1,
    selectedLineIndex: null,
    zoomLevel: 100,
    isDrawingMode: false,
    isDrawing: false,
    drawStart: null,
    deviceInfoExpanded: false,
    // LLM state
    llmStatus: {
        ollamaAvailable: false,
        models: [],
        selectedModel: null
    },
    llmResult: null,
    // Schema management state
    schemas: [],
    selectedSchemaId: null,
    schemaYamlDirty: false
};

// DOM elements
const elements = {
    fileInput: document.getElementById('fileInput'),
    uploadArea: document.getElementById('uploadArea'),
    imageList: document.getElementById('imageList'),
    normalizeBtn: document.getElementById('normalizeBtn'),
    recognizeBtn: document.getElementById('recognizeBtn'),
    addBboxBtn: document.getElementById('addBboxBtn'),
    saveBtn: document.getElementById('saveBtn'),
    placeholder: document.getElementById('placeholder'),
    zoomWrapper: document.getElementById('zoomWrapper'),
    imageContainer: document.getElementById('imageContainer'),
    mainImage: document.getElementById('mainImage'),
    textOverlay: document.getElementById('textOverlay'),
    jsonOutput: document.getElementById('jsonOutput'),
    zoomControls: document.getElementById('zoomControls'),
    zoomSlider: document.getElementById('zoomSlider'),
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    zoomValue: document.getElementById('zoomValue'),
    documentClassContainer: document.getElementById('documentClassContainer'),
    documentClassValue: document.getElementById('documentClassValue'),
    documentClassConfidence: document.getElementById('documentClassConfidence'),
    // Device info elements
    deviceInfoStatus: document.getElementById('deviceInfoStatus'),
    deviceInfoDetails: document.getElementById('deviceInfoDetails'),
    deviceInfoToggle: document.getElementById('deviceInfoToggle'),
    pytorchVersion: document.getElementById('pytorchVersion'),
    pytorchCudaBuild: document.getElementById('pytorchCudaBuild'),
    pytorchCudaRow: document.getElementById('pytorchCudaRow'),
    accelerationType: document.getElementById('accelerationType'),
    selectedDevice: document.getElementById('selectedDevice'),
    cudaVersion: document.getElementById('cudaVersion'),
    cudaVersionRow: document.getElementById('cudaVersionRow'),
    gpuCount: document.getElementById('gpuCount'),
    gpuCountRow: document.getElementById('gpuCountRow'),
    gpuName: document.getElementById('gpuName'),
    gpuNameRow: document.getElementById('gpuNameRow'),
    gpuMemory: document.getElementById('gpuMemory'),
    gpuMemoryRow: document.getElementById('gpuMemoryRow'),
    suryaDevice: document.getElementById('suryaDevice'),
    clipDevice: document.getElementById('clipDevice'),
    // LLM elements
    jsonTab: document.getElementById('jsonTab'),
    llmTab: document.getElementById('llmTab'),
    ollamaStatus: document.getElementById('ollamaStatus'),
    ollamaStatusText: document.getElementById('ollamaStatusText'),
    llmModelSelect: document.getElementById('llmModelSelect'),
    pullModelBtn: document.getElementById('pullModelBtn'),
    processLlmBtn: document.getElementById('processLlmBtn'),
    llmPlaceholder: document.getElementById('llmPlaceholder'),
    llmResult: document.getElementById('llmResult'),
    llmEnhancedText: document.getElementById('llmEnhancedText'),
    llmMedicationsSection: document.getElementById('llmMedicationsSection'),
    llmMedications: document.getElementById('llmMedications'),
    llmModelUsed: document.getElementById('llmModelUsed'),
    llmLoading: document.getElementById('llmLoading')
};

// Initialization
document.addEventListener('DOMContentLoaded', init);

function init() {
    // File upload
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Buttons
    elements.normalizeBtn.addEventListener('click', normalizeImage);
    elements.recognizeBtn.addEventListener('click', recognizeImage);
    elements.addBboxBtn.addEventListener('click', toggleDrawingMode);
    elements.saveBtn.addEventListener('click', saveJson);
    
    // Update scale when image size changes
    elements.mainImage.addEventListener('load', updateImageScale);
    
    // Zoom controls
    elements.zoomSlider.addEventListener('input', handleZoomSlider);
    elements.zoomInBtn.addEventListener('click', () => setZoom(state.zoomLevel + 10));
    elements.zoomOutBtn.addEventListener('click', () => setZoom(state.zoomLevel - 10));
    
    // Keyboard zoom (Ctrl++ / Ctrl+-)
    document.addEventListener('keydown', handleKeyboardZoom);
    
    // Fetch device info on load
    console.log('Initializing app, fetching device info...');
    fetchDeviceInfo();
    
    // LLM controls
    if (elements.pullModelBtn) {
        elements.pullModelBtn.addEventListener('click', pullSelectedModel);
    }
    if (elements.processLlmBtn) {
        elements.processLlmBtn.addEventListener('click', processWithLlm);
    }
    if (elements.llmModelSelect) {
        elements.llmModelSelect.addEventListener('change', handleModelSelect);
    }
    
    // Check LLM status
    checkLlmStatus();
    
    // Initialize schema management
    initSchemaManagement();
}

// === File upload ===

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

async function uploadFiles(files) {
    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        
        // Add images to list
        for (const img of data.images) {
            state.images.push(img);
            addImageToList(img);
        }
        
        // Select first uploaded if nothing selected
        if (!state.selectedImageId && data.images.length > 0) {
            selectImage(data.images[0].id);
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading files');
    }
    
    // Reset input
    elements.fileInput.value = '';
}

function addImageToList(image) {
    const item = document.createElement('div');
    item.className = 'image-item';
    item.dataset.id = image.id;
    item.innerHTML = `
        <img src="/api/image/${image.id}" alt="${image.filename}">
        <span class="filename">${image.filename}</span>
    `;
    item.addEventListener('click', () => selectImage(image.id));
    elements.imageList.appendChild(item);
}

function selectImage(imageId) {
    state.selectedImageId = imageId;
    state.ocrResult = null;
    state.selectedLineIndex = null;
    
    // Update list UI
    document.querySelectorAll('.image-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.id === imageId);
    });
    
    // Show image and zoom controls
    elements.placeholder.style.display = 'none';
    elements.zoomWrapper.style.display = 'flex';
    elements.zoomControls.style.display = 'flex';
    elements.mainImage.src = `/api/image/${imageId}?t=${Date.now()}`;
    
    // Reset zoom
    setZoom(100);
    
    // Clear overlay and JSON
    elements.textOverlay.innerHTML = '';
    elements.jsonOutput.textContent = JSON.stringify({ text_lines: [], image_bbox: [] }, null, 2);
    
    // Enable buttons
    elements.normalizeBtn.disabled = false;
    elements.recognizeBtn.disabled = false;
    elements.saveBtn.disabled = true;
}

// === Image processing ===

async function normalizeImage() {
    if (!state.selectedImageId) return;
    
    elements.normalizeBtn.classList.add('loading');
    elements.normalizeBtn.disabled = true;
    
    try {
        const response = await fetch('/api/normalize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: state.selectedImageId })
        });
        
        if (!response.ok) {
            throw new Error('Normalize failed');
        }
        
        const data = await response.json();
        
        // Update image
        elements.mainImage.src = `/api/image/${state.selectedImageId}?t=${Date.now()}`;
        
        // Update preview in list
        const listItem = document.querySelector(`.image-item[data-id="${state.selectedImageId}"] img`);
        if (listItem) {
            listItem.src = `/api/image/${state.selectedImageId}?t=${Date.now()}`;
        }
        
        // Clear overlay on normalization
        elements.textOverlay.innerHTML = '';
        state.ocrResult = null;
        elements.saveBtn.disabled = true;
        
        console.log(`Normalized by ${data.angle.toFixed(2)}°`);
        
    } catch (error) {
        console.error('Normalize error:', error);
        alert('Normalization error');
    } finally {
        elements.normalizeBtn.classList.remove('loading');
        elements.normalizeBtn.disabled = false;
    }
}

async function recognizeImage() {
    if (!state.selectedImageId) return;
    
    elements.recognizeBtn.classList.add('loading');
    elements.recognizeBtn.disabled = true;
    
    try {
        const response = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: state.selectedImageId })
        });
        
        if (!response.ok) {
            throw new Error('Recognition failed');
        }
        
        const data = await response.json();
        state.ocrResult = data;
        
        // Display JSON
        updateJsonOutput();
        
        // Create overlay
        renderTextOverlay();
        
        // Display document class
        updateDocumentClass();
        
        // Enable buttons
        elements.saveBtn.disabled = false;
        elements.addBboxBtn.disabled = false;
        
        // Update LLM process button state
        updateProcessButtonState();
        
        // Refresh device info (models are now initialized)
        fetchDeviceInfo();
        
    } catch (error) {
        console.error('Recognition error:', error);
        alert('Recognition error');
    } finally {
        elements.recognizeBtn.classList.remove('loading');
        elements.recognizeBtn.disabled = false;
    }
}

// === Display results ===

function updateImageScale() {
    const img = elements.mainImage;
    if (img.naturalWidth > 0) {
        state.imageScale = img.clientWidth / img.naturalWidth;
    }
    
    // Redraw overlay if results exist
    if (state.ocrResult) {
        renderTextOverlay();
    }
}

function renderTextOverlay() {
    elements.textOverlay.innerHTML = '';
    
    if (!state.ocrResult || !state.ocrResult.text_lines) return;
    
    const scale = state.imageScale;
    
    state.ocrResult.text_lines.forEach((line, index) => {
        if (!line.bbox || line.bbox.length < 4) return;
        
        const [x1, y1, x2, y2] = line.bbox;
        
        const block = document.createElement('div');
        block.className = 'bbox-block ' + getConfidenceClass(line.confidence);
        block.dataset.index = index;
        
        block.style.left = `${x1 * scale}px`;
        block.style.top = `${y1 * scale}px`;
        block.style.width = `${(x2 - x1) * scale}px`;
        block.style.height = `${(y2 - y1) * scale}px`;
        
        // Click - select and highlight in JSON
        block.addEventListener('click', (e) => {
            e.stopPropagation();
            selectLine(index);
        });
        
        // Double-click - open editor
        block.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            openEditor(index, block);
        });
        
        elements.textOverlay.appendChild(block);
    });
}

function selectLine(index) {
    state.selectedLineIndex = index;
    
    // Remove selection from all bbox
    document.querySelectorAll('.bbox-block').forEach(b => b.classList.remove('selected'));
    
    // Select current
    const block = document.querySelector(`.bbox-block[data-index="${index}"]`);
    if (block) {
        block.classList.add('selected');
    }
    
    // Highlight in JSON
    highlightJsonLine(index);
}

function highlightJsonLine(index) {
    if (!state.ocrResult || !state.ocrResult.text_lines[index]) return;
    
    const line = state.ocrResult.text_lines[index];
    const fullJson = JSON.stringify(state.ocrResult, null, 2);
    const lineJson = JSON.stringify(line, null, 2);
    
    // Find position of this line in JSON
    const lineText = `"text": "${line.text}"`;
    
    // Create HTML with highlighting
    const escapedLineJson = lineJson.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
    const escapedFullJson = fullJson.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
    
    // Simple highlighting - show selected line separately
    elements.jsonOutput.innerHTML = `<span class="json-highlight">// Selected [${index}]:\n${escapedLineJson}</span>\n\n${escapedFullJson}`;
}

function openEditor(index, bboxBlock) {
    const line = state.ocrResult.text_lines[index];
    if (!line) return;
    
    // Check if editor is already open
    const existingEditor = bboxBlock.querySelector('.inline-editor');
    if (existingEditor) return;
    
    // Save original dimensions
    const originalWidth = bboxBlock.style.width;
    const originalHeight = bboxBlock.style.height;
    
    // Create editor
    const editor = document.createElement('input');
    editor.type = 'text';
    editor.className = 'inline-editor';
    editor.value = line.text;
    
    // Save on Enter or blur
    const saveEdit = () => {
        const newText = editor.value;
        state.ocrResult.text_lines[index].text = newText;
        updateJsonOutput();
        highlightJsonLine(index);
        editor.remove();
        bboxBlock.classList.remove('editing');
        // Restore original dimensions
        bboxBlock.style.width = originalWidth;
        bboxBlock.style.height = originalHeight;
    };
    
    const cancelEdit = () => {
        editor.remove();
        bboxBlock.classList.remove('editing');
        // Restore original dimensions
        bboxBlock.style.width = originalWidth;
        bboxBlock.style.height = originalHeight;
    };
    
    editor.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            saveEdit();
        } else if (e.key === 'Escape') {
            cancelEdit();
        }
    });
    
    editor.addEventListener('blur', saveEdit);
    
    bboxBlock.classList.add('editing');
    bboxBlock.appendChild(editor);
    
    // Expand bbox to show full text
    const textWidth = measureTextWidth(line.text, '14px -apple-system, BlinkMacSystemFont, sans-serif');
    const minWidth = Math.max(parseFloat(originalWidth), textWidth + 20);
    bboxBlock.style.width = `${minWidth}px`;
    bboxBlock.style.height = 'auto';
    bboxBlock.style.minHeight = '28px';
    
    editor.focus();
    editor.select();
}

function measureTextWidth(text, font) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = font;
    return ctx.measureText(text).width;
}

function getConfidenceClass(confidence) {
    if (confidence < 0.8) {
        return 'confidence-low';
    } else if (confidence < 0.93) {
        return 'confidence-medium';
    } else {
        return 'confidence-high';
    }
}

function updateJsonOutput() {
    if (state.selectedLineIndex !== null) {
        highlightJsonLine(state.selectedLineIndex);
    } else {
        elements.jsonOutput.textContent = JSON.stringify(state.ocrResult, null, 2);
    }
}

function updateDocumentClass() {
    if (!state.ocrResult || !state.ocrResult.document_class) {
        elements.documentClassContainer.style.display = 'none';
        return;
    }
    
    const docClass = state.ocrResult.document_class;
    const className = docClass.class || 'Undetected';
    const confidence = docClass.confidence || 0;
    
    // Show container
    elements.documentClassContainer.style.display = 'flex';
    
    // Set text
    elements.documentClassValue.textContent = className;
    elements.documentClassConfidence.textContent = confidence > 0 ? `(${Math.round(confidence * 100)}%)` : '';
    
    // Set CSS class for color
    elements.documentClassValue.className = 'document-class-value ' + getDocumentClassCss(className);
}

function getDocumentClassCss(className) {
    const classMap = {
        'Receipt': 'class-receipt',
        'Medication Prescription': 'class-prescription',
        'Form': 'class-form',
        'Contract': 'class-contract',
        'Undetected': 'class-undetected'
    };
    return classMap[className] || 'class-undetected';
}

// === Save ===

function saveJson() {
    if (!state.ocrResult) return;
    
    const json = JSON.stringify(state.ocrResult, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr_result_${state.selectedImageId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Handle window resize
window.addEventListener('resize', () => {
    if (state.ocrResult) {
        updateImageScale();
    }
});

// === Zoom functions ===

function setZoom(level) {
    // Limit zoom to 25-300%
    level = Math.max(25, Math.min(300, level));
    state.zoomLevel = level;
    
    // Update UI
    elements.zoomSlider.value = level;
    elements.zoomValue.textContent = `${level}%`;
    
    // Apply transform to container
    const scale = level / 100;
    elements.imageContainer.style.transform = `scale(${scale})`;
    
    // Redraw overlay with zoom
    if (state.ocrResult) {
        renderTextOverlay();
    }
}

function handleZoomSlider(e) {
    setZoom(parseInt(e.target.value, 10));
}

function handleKeyboardZoom(e) {
    // Ctrl++ or Ctrl+=
    if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '=' || e.key === 'Equal')) {
        e.preventDefault();
        setZoom(state.zoomLevel + 10);
    }
    // Ctrl+-
    else if ((e.ctrlKey || e.metaKey) && (e.key === '-' || e.key === 'Minus')) {
        e.preventDefault();
        setZoom(state.zoomLevel - 10);
    }
    // Ctrl+0 - reset zoom
    else if ((e.ctrlKey || e.metaKey) && e.key === '0') {
        e.preventDefault();
        setZoom(100);
    }
    // Escape - exit drawing mode
    else if (e.key === 'Escape' && state.isDrawingMode) {
        toggleDrawingMode();
    }
}

// === BBox drawing mode ===

function toggleDrawingMode() {
    state.isDrawingMode = !state.isDrawingMode;
    elements.addBboxBtn.classList.toggle('active', state.isDrawingMode);
    elements.imageContainer.classList.toggle('drawing-mode', state.isDrawingMode);
    
    if (state.isDrawingMode) {
        createDrawingOverlay();
    } else {
        removeDrawingOverlay();
    }
}

function createDrawingOverlay() {
    // Remove existing overlay if present
    removeDrawingOverlay();
    
    const overlay = document.createElement('div');
    overlay.className = 'drawing-overlay';
    overlay.id = 'drawingOverlay';
    
    overlay.addEventListener('mousedown', startDrawing);
    overlay.addEventListener('mousemove', updateDrawing);
    overlay.addEventListener('mouseup', finishDrawing);
    overlay.addEventListener('mouseleave', cancelDrawing);
    
    elements.imageContainer.appendChild(overlay);
}

function removeDrawingOverlay() {
    const overlay = document.getElementById('drawingOverlay');
    if (overlay) {
        overlay.remove();
    }
    // Remove temporary rectangle
    const rect = document.getElementById('drawingRect');
    if (rect) {
        rect.remove();
    }
}

function startDrawing(e) {
    if (!state.isDrawingMode) return;
    
    state.isDrawing = true;
    const rect = e.target.getBoundingClientRect();
    const scale = state.zoomLevel / 100;
    
    state.drawStart = {
        x: (e.clientX - rect.left) / scale,
        y: (e.clientY - rect.top) / scale
    };
    
    // Create temporary rectangle
    const drawRect = document.createElement('div');
    drawRect.className = 'drawing-rect';
    drawRect.id = 'drawingRect';
    drawRect.style.left = `${state.drawStart.x}px`;
    drawRect.style.top = `${state.drawStart.y}px`;
    drawRect.style.width = '0px';
    drawRect.style.height = '0px';
    elements.imageContainer.appendChild(drawRect);
}

function updateDrawing(e) {
    if (!state.isDrawing || !state.drawStart) return;
    
    const overlay = e.target;
    const rect = overlay.getBoundingClientRect();
    const scale = state.zoomLevel / 100;
    
    const currentX = (e.clientX - rect.left) / scale;
    const currentY = (e.clientY - rect.top) / scale;
    
    const drawRect = document.getElementById('drawingRect');
    if (!drawRect) return;
    
    const x = Math.min(state.drawStart.x, currentX);
    const y = Math.min(state.drawStart.y, currentY);
    const width = Math.abs(currentX - state.drawStart.x);
    const height = Math.abs(currentY - state.drawStart.y);
    
    drawRect.style.left = `${x}px`;
    drawRect.style.top = `${y}px`;
    drawRect.style.width = `${width}px`;
    drawRect.style.height = `${height}px`;
}

function cancelDrawing() {
    if (state.isDrawing) {
        state.isDrawing = false;
        state.drawStart = null;
        const drawRect = document.getElementById('drawingRect');
        if (drawRect) {
            drawRect.remove();
        }
    }
}

async function finishDrawing(e) {
    if (!state.isDrawing || !state.drawStart) return;
    
    const overlay = e.target;
    const rect = overlay.getBoundingClientRect();
    const scale = state.zoomLevel / 100;
    
    const endX = (e.clientX - rect.left) / scale;
    const endY = (e.clientY - rect.top) / scale;
    
    // Calculate bbox in image coordinates
    const imgWidth = elements.mainImage.naturalWidth;
    const imgHeight = elements.mainImage.naturalHeight;
    const displayWidth = elements.mainImage.width;
    const displayHeight = elements.mainImage.height;
    
    const scaleX = imgWidth / displayWidth;
    const scaleY = imgHeight / displayHeight;
    
    const x1 = Math.min(state.drawStart.x, endX) * scaleX;
    const y1 = Math.min(state.drawStart.y, endY) * scaleY;
    const x2 = Math.max(state.drawStart.x, endX) * scaleX;
    const y2 = Math.max(state.drawStart.y, endY) * scaleY;
    
    // Minimum bbox size
    if (x2 - x1 < 10 || y2 - y1 < 10) {
        cancelDrawing();
        return;
    }
    
    state.isDrawing = false;
    state.drawStart = null;
    
    // Remove temporary rectangle
    const drawRect = document.getElementById('drawingRect');
    if (drawRect) {
        drawRect.remove();
    }
    
    // Exit drawing mode
    toggleDrawingMode();
    
    // Process new bbox
    await processNewBbox([x1, y1, x2, y2]);
}

async function processNewBbox(bbox) {
    if (!state.ocrResult) {
        state.ocrResult = { text_lines: [], image_bbox: [] };
    }
    
    // Remove overlapping bboxes from result
    const remainingLines = state.ocrResult.text_lines.filter(line => {
        return !bboxOverlaps(line.bbox, bbox);
    });
    
    state.ocrResult.text_lines = remainingLines;
    
    // Show loading indicator
    elements.addBboxBtn.classList.add('loading');
    elements.addBboxBtn.disabled = true;
    
    try {
        // Recognize text in new region
        const response = await fetch('/api/recognize-region', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.selectedImageId,
                bbox: bbox
            })
        });
        
        if (!response.ok) {
            throw new Error('Recognition failed');
        }
        
        const result = await response.json();
        
        // Add new recognized lines
        if (result.text_lines && result.text_lines.length > 0) {
            state.ocrResult.text_lines.push(...result.text_lines);
        }
        
        // Sort lines by position (top to bottom, left to right)
        state.ocrResult.text_lines.sort((a, b) => {
            const yDiff = a.bbox[1] - b.bbox[1];
            if (Math.abs(yDiff) > 10) return yDiff;
            return a.bbox[0] - b.bbox[0];
        });
        
        // Update UI
        renderTextOverlay();
        updateJsonOutput();
        elements.saveBtn.disabled = false;
        
    } catch (error) {
        console.error('Error recognizing region:', error);
        alert('Error recognizing region');
    } finally {
        elements.addBboxBtn.classList.remove('loading');
        elements.addBboxBtn.disabled = false;
    }
}

function bboxOverlaps(bbox1, bbox2) {
    // Check if two bboxes overlap
    // bbox format: [x1, y1, x2, y2]
    const [ax1, ay1, ax2, ay2] = bbox1;
    const [bx1, by1, bx2, by2] = bbox2;
    
    // Check intersection
    const overlapX = ax1 < bx2 && ax2 > bx1;
    const overlapY = ay1 < by2 && ay2 > by1;
    
    if (!overlapX || !overlapY) return false;
    
    // Calculate intersection area
    const intersectX1 = Math.max(ax1, bx1);
    const intersectY1 = Math.max(ay1, by1);
    const intersectX2 = Math.min(ax2, bx2);
    const intersectY2 = Math.min(ay2, by2);
    
    const intersectArea = (intersectX2 - intersectX1) * (intersectY2 - intersectY1);
    const bbox1Area = (ax2 - ax1) * (ay2 - ay1);
    
    // Consider overlap if intersection > 30% of bbox1
    return intersectArea > bbox1Area * 0.3;
}

// === Device Info ===

async function fetchDeviceInfo() {
    console.log('fetchDeviceInfo called');
    try {
        const response = await fetch('/api/device-info');
        console.log('Device info response status:', response.status);
        if (!response.ok) {
            throw new Error('Failed to fetch device info: ' + response.status);
        }
        const info = await response.json();
        console.log('Device info received:', info);
        updateDeviceInfoUI(info);
        console.log('Device info UI updated');
    } catch (error) {
        console.error('Error fetching device info:', error);
        if (elements.deviceInfoStatus) {
            elements.deviceInfoStatus.textContent = 'Error';
            elements.deviceInfoStatus.className = 'device-info-status status-error';
        }
    }
}

function updateDeviceInfoUI(info) {
    if (!elements.deviceInfoStatus) {
        console.error('Device info elements not found in DOM');
        return;
    }
    
    // Update PyTorch version
    if (elements.pytorchVersion) {
        elements.pytorchVersion.textContent = info.pytorch_version || '-';
    }
    
    // Update PyTorch CUDA build status
    if (elements.pytorchCudaBuild) {
        if (info.pytorch_cuda_built) {
            elements.pytorchCudaBuild.textContent = info.pytorch_cuda_version || 'Yes';
            elements.pytorchCudaBuild.className = 'device-value status-gpu';
        } else {
            elements.pytorchCudaBuild.textContent = 'No (CPU only)';
            elements.pytorchCudaBuild.className = 'device-value status-error';
        }
    }
    
    // Update acceleration type
    if (elements.accelerationType) {
        elements.accelerationType.textContent = info.acceleration_type || 'Unknown';
        const isGpu = info.cuda_available || info.mps_available;
        elements.accelerationType.className = 'device-value ' + (isGpu ? 'status-gpu' : 'status-cpu');
    }
    
    // Update selected device
    if (elements.selectedDevice) {
        elements.selectedDevice.textContent = info.selected_device || '-';
        elements.selectedDevice.className = 'device-value ' + getDeviceStatusClass(info.selected_device || 'cpu');
    }
    
    // Update CUDA/NVIDIA info (show only if CUDA available)
    if (info.cuda_available) {
        if (elements.cudaVersionRow) elements.cudaVersionRow.style.display = 'flex';
        if (elements.cudaVersion) elements.cudaVersion.textContent = info.cuda_version || '-';
        if (elements.gpuCountRow) elements.gpuCountRow.style.display = 'flex';
        if (elements.gpuCount) elements.gpuCount.textContent = info.gpu_count || '-';
        if (elements.gpuNameRow) elements.gpuNameRow.style.display = 'flex';
        if (elements.gpuName) elements.gpuName.textContent = info.gpu_name || '-';
        if (info.gpu_memory_total) {
            if (elements.gpuMemoryRow) elements.gpuMemoryRow.style.display = 'flex';
            if (elements.gpuMemory) elements.gpuMemory.textContent = info.gpu_memory_total;
        }
    }
    
    // Update Surya device
    if (elements.suryaDevice) {
        elements.suryaDevice.textContent = info.surya_device || 'Pending';
        elements.suryaDevice.className = 'device-value ' + (info.surya_device ? getDeviceStatusClass(info.surya_device) : 'status-pending');
    }
    
    // Update CLIP device
    if (elements.clipDevice) {
        elements.clipDevice.textContent = info.clip_device || 'Pending';
        elements.clipDevice.className = 'device-value ' + (info.clip_device ? getDeviceStatusClass(info.clip_device) : 'status-pending');
    }
    
    // Hide note if models are initialized
    const noteEl = document.getElementById('deviceInfoNote');
    if (noteEl && info.surya_device && info.clip_device) {
        noteEl.style.display = 'none';
    }
    
    // Update status summary in header
    const hasGpu = info.cuda_available || info.mps_available;
    const activeDevice = info.surya_device || info.clip_device;
    if (activeDevice && (activeDevice.includes('cuda') || activeDevice.includes('mps'))) {
        if (info.cuda_available) {
            elements.deviceInfoStatus.textContent = 'CUDA Active';
        } else {
            elements.deviceInfoStatus.textContent = 'MPS Active';
        }
        elements.deviceInfoStatus.className = 'device-info-status status-gpu';
    } else if (hasGpu) {
        if (info.cuda_available) {
            elements.deviceInfoStatus.textContent = 'CUDA Ready';
        } else {
            elements.deviceInfoStatus.textContent = 'MPS Ready';
        }
        elements.deviceInfoStatus.className = 'device-info-status status-available';
    } else {
        elements.deviceInfoStatus.textContent = 'CPU Only';
        elements.deviceInfoStatus.className = 'device-info-status status-cpu';
    }
}

function getDeviceStatusClass(device) {
    if (device.includes('cuda') || device.includes('mps')) {
        return 'status-gpu';
    }
    return 'status-cpu';
}

function toggleDeviceInfo() {
    state.deviceInfoExpanded = !state.deviceInfoExpanded;
    elements.deviceInfoDetails.style.display = state.deviceInfoExpanded ? 'block' : 'none';
    elements.deviceInfoToggle.textContent = state.deviceInfoExpanded ? '▲' : '▼';
    
    // Refresh device info when expanding
    if (state.deviceInfoExpanded) {
        fetchDeviceInfo();
    }
}

// =============================================================================
// LLM POST-PROCESSING
// =============================================================================

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update tab content
    if (elements.jsonTab) {
        elements.jsonTab.style.display = tabName === 'json' ? 'flex' : 'none';
        elements.jsonTab.classList.toggle('active', tabName === 'json');
    }
    if (elements.llmTab) {
        elements.llmTab.style.display = tabName === 'llm' ? 'flex' : 'none';
        elements.llmTab.classList.toggle('active', tabName === 'llm');
    }
}

async function checkLlmStatus() {
    try {
        const response = await fetch('/api/llm/status');
        if (!response.ok) throw new Error('Failed to check LLM status');
        
        const data = await response.json();
        state.llmStatus.ollamaAvailable = data.ollama_available;
        state.llmStatus.models = data.models;
        state.llmStatus.acceleration = data.acceleration;
        state.llmStatus.accelerationDetails = data.acceleration_details;
        state.llmStatus.version = data.version;
        
        updateLlmStatusUI();
    } catch (error) {
        console.error('LLM status check failed:', error);
        state.llmStatus.ollamaAvailable = false;
        updateLlmStatusUI();
    }
}

function updateLlmStatusUI() {
    const { ollamaAvailable, models, acceleration, accelerationDetails } = state.llmStatus;
    
    // Update status indicator
    if (elements.ollamaStatus) {
        elements.ollamaStatus.textContent = ollamaAvailable ? '🟢' : '🔴';
        elements.ollamaStatus.className = 'status-indicator ' + (ollamaAvailable ? 'online' : 'offline');
    }
    if (elements.ollamaStatusText) {
        elements.ollamaStatusText.textContent = ollamaAvailable ? 'Ollama Connected' : 'Ollama Offline';
    }
    
    // Update acceleration display
    const accelContainer = document.getElementById('llmAcceleration');
    const accelIcon = document.getElementById('accelIcon');
    const accelText = document.getElementById('accelText');
    
    if (accelContainer && accelIcon && accelText) {
        accelContainer.className = 'llm-acceleration';
        
        if (acceleration === 'metal') {
            accelContainer.classList.add('metal');
            accelIcon.textContent = '🍎';
            accelText.textContent = accelerationDetails || 'Metal (Apple Silicon)';
        } else if (acceleration === 'cuda') {
            accelContainer.classList.add('cuda');
            accelIcon.textContent = '🟢';
            accelText.textContent = accelerationDetails || 'CUDA (NVIDIA GPU)';
        } else if (acceleration === 'rocm') {
            accelContainer.classList.add('cuda');
            accelIcon.textContent = '🔴';
            accelText.textContent = accelerationDetails || 'ROCm (AMD GPU)';
        } else {
            accelContainer.classList.add('cpu');
            accelIcon.textContent = '💻';
            accelText.textContent = accelerationDetails || 'CPU Only';
        }
    }
    
    // Update model select
    if (elements.llmModelSelect) {
        elements.llmModelSelect.innerHTML = '';
        elements.llmModelSelect.disabled = !ollamaAvailable;
        
        if (ollamaAvailable && models.length > 0) {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name + (model.available ? '' : ' (not pulled)');
                option.dataset.available = model.available;
                elements.llmModelSelect.appendChild(option);
            });
            
            // Select first available model
            const firstAvailable = models.find(m => m.available);
            if (firstAvailable) {
                elements.llmModelSelect.value = firstAvailable.id;
                state.llmStatus.selectedModel = firstAvailable.id;
            }
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = ollamaAvailable ? 'No models' : 'Ollama offline';
            elements.llmModelSelect.appendChild(option);
        }
        
        handleModelSelect();
    }
}

function handleModelSelect() {
    if (!elements.llmModelSelect) return;
    
    const selectedOption = elements.llmModelSelect.selectedOptions[0];
    const isAvailable = selectedOption?.dataset.available === 'true';
    
    state.llmStatus.selectedModel = elements.llmModelSelect.value;
    
    // Update pull button
    if (elements.pullModelBtn) {
        elements.pullModelBtn.disabled = !state.llmStatus.ollamaAvailable || isAvailable;
        elements.pullModelBtn.title = isAvailable ? 'Model already available' : 'Pull model from registry';
    }
    
    // Update process button
    updateProcessButtonState();
}

function updateProcessButtonState() {
    if (!elements.processLlmBtn) return;
    
    const hasOcr = state.ocrResult && state.ocrResult.text_lines && state.ocrResult.text_lines.length > 0;
    const hasModel = state.llmStatus.ollamaAvailable && state.llmStatus.selectedModel;
    const selectedOption = elements.llmModelSelect?.selectedOptions[0];
    const modelAvailable = selectedOption?.dataset.available === 'true';
    
    elements.processLlmBtn.disabled = !hasOcr || !hasModel || !modelAvailable;
}

async function pullSelectedModel() {
    const modelId = state.llmStatus.selectedModel;
    if (!modelId) return;
    
    elements.pullModelBtn.disabled = true;
    elements.pullModelBtn.textContent = '⏳';
    
    // Show progress bar
    const progressContainer = document.getElementById('pullProgress');
    const progressFill = document.getElementById('pullProgressFill');
    const progressText = document.getElementById('pullProgressText');
    
    if (progressContainer) {
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Starting download...';
    }
    
    try {
        // Use streaming endpoint for progress updates
        const eventSource = new EventSource(`/api/llm/pull/${encodeURIComponent(modelId)}/stream`);
        
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.status === 'already_available') {
                    progressText.textContent = 'Model already available';
                    progressFill.style.width = '100%';
                    eventSource.close();
                    setTimeout(() => finishPull(true), 500);
                    return;
                }
                
                if (data.status === 'complete') {
                    progressText.textContent = 'Download complete!';
                    progressFill.style.width = '100%';
                    eventSource.close();
                    setTimeout(() => finishPull(true), 500);
                    return;
                }
                
                if (data.status === 'error') {
                    progressText.textContent = 'Error: ' + (data.error || 'Unknown error');
                    eventSource.close();
                    setTimeout(() => finishPull(false), 2000);
                    return;
                }
                
                // Update progress
                if (data.total && data.completed) {
                    const percent = Math.round((data.completed / data.total) * 100);
                    progressFill.style.width = percent + '%';
                    
                    // Format size
                    const completedMB = (data.completed / 1024 / 1024).toFixed(1);
                    const totalMB = (data.total / 1024 / 1024).toFixed(1);
                    progressText.textContent = `Downloading: ${completedMB} MB / ${totalMB} MB (${percent}%)`;
                } else if (data.status) {
                    progressText.textContent = data.status;
                }
            } catch (e) {
                console.error('Error parsing progress:', e);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            eventSource.close();
            finishPull(false);
        };
        
    } catch (error) {
        console.error('Model pull failed:', error);
        alert('Failed to pull model: ' + error.message);
        finishPull(false);
    }
    
    async function finishPull(success) {
        elements.pullModelBtn.textContent = '⬇️';
        
        if (progressContainer) {
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 1000);
        }
        
        // Refresh status
        await checkLlmStatus();
        
        if (!success) {
            alert('Failed to pull model');
        }
    }
}

async function processWithLlm() {
    if (!state.ocrResult || !state.ocrResult.text_lines) {
        alert('No OCR results to process');
        return;
    }
    
    const modelId = state.llmStatus.selectedModel;
    if (!modelId) {
        alert('No model selected');
        return;
    }
    
    // Get document type from OCR result
    const docType = state.ocrResult.document_class?.type_id || 
                    state.ocrResult.document_class?.class?.toLowerCase() || 
                    null;
    
    // Combine text lines into single text
    const fullText = state.ocrResult.text_lines
        .map(line => line.text)
        .join('\n');
    
    // Show loading
    showLlmLoading(true);
    
    try {
        const response = await fetch('/api/llm/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: fullText,
                model: modelId,
                document_type: docType
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'LLM processing failed');
        }
        
        const result = await response.json();
        state.llmResult = result;
        
        displayLlmResult(result);
        
    } catch (error) {
        console.error('LLM processing failed:', error);
        alert('LLM processing failed: ' + error.message);
        showLlmLoading(false);
    }
}

function showLlmLoading(show) {
    if (elements.llmPlaceholder) elements.llmPlaceholder.style.display = 'none';
    if (elements.llmResult) elements.llmResult.style.display = show ? 'none' : 'block';
    if (elements.llmLoading) elements.llmLoading.style.display = show ? 'flex' : 'none';
}

function displayLlmResult(result) {
    showLlmLoading(false);
    
    if (!result.success) {
        if (elements.llmPlaceholder) {
            elements.llmPlaceholder.innerHTML = `<p>Error: ${result.error || 'Processing failed'}</p>`;
            elements.llmPlaceholder.style.display = 'flex';
        }
        if (elements.llmResult) elements.llmResult.style.display = 'none';
        return;
    }
    
    if (elements.llmResult) elements.llmResult.style.display = 'block';
    if (elements.llmPlaceholder) elements.llmPlaceholder.style.display = 'none';
    
    // Display document type badge
    const docTypeBadge = document.getElementById('llmDocTypeBadge');
    if (docTypeBadge) {
        docTypeBadge.textContent = result.document_type_name || result.document_type || '';
    }
    
    // Display extracted fields
    const fieldsContainer = document.getElementById('llmExtractedFields');
    if (fieldsContainer && result.extracted_fields) {
        const requiredFields = result.required_fields || [];
        fieldsContainer.innerHTML = '';
        
        for (const [fieldName, fieldValue] of Object.entries(result.extracted_fields)) {
            const fieldItem = document.createElement('div');
            fieldItem.className = 'field-item';
            
            const isRequired = requiredFields.includes(fieldName);
            const displayName = fieldName.replace(/_/g, ' ');
            
            const nameEl = document.createElement('div');
            nameEl.className = 'field-name' + (isRequired ? ' required' : '');
            nameEl.textContent = displayName;
            fieldItem.appendChild(nameEl);
            
            const valueEl = document.createElement('div');
            valueEl.className = 'field-value';
            
            if (fieldValue === null || fieldValue === undefined) {
                valueEl.className += ' empty';
                valueEl.textContent = 'Not found';
            } else if (Array.isArray(fieldValue)) {
                valueEl.className += ' list-value';
                if (fieldValue.length === 0) {
                    valueEl.className += ' empty';
                    valueEl.textContent = 'None';
                } else {
                    fieldValue.forEach(item => {
                        const itemEl = document.createElement('div');
                        itemEl.className = 'list-item';
                        itemEl.textContent = typeof item === 'object' ? JSON.stringify(item) : item;
                        valueEl.appendChild(itemEl);
                    });
                }
            } else {
                valueEl.textContent = fieldValue;
            }
            
            fieldItem.appendChild(valueEl);
            fieldsContainer.appendChild(fieldItem);
        }
    }
    
    // Display corrected text
    if (elements.llmEnhancedText) {
        elements.llmEnhancedText.textContent = result.corrected_text || result.processed_text || '';
    }
    
    // Display confidence notes if present
    const confidenceSection = document.getElementById('llmConfidenceSection');
    const confidenceNotes = document.getElementById('llmConfidenceNotes');
    if (result.confidence_notes && result.confidence_notes.trim()) {
        if (confidenceSection) confidenceSection.style.display = 'block';
        if (confidenceNotes) confidenceNotes.textContent = result.confidence_notes;
    } else {
        if (confidenceSection) confidenceSection.style.display = 'none';
    }
    
    // Display model used
    if (elements.llmModelUsed) {
        elements.llmModelUsed.textContent = result.model || 'Unknown';
    }
    
    // Display document type
    const docTypeEl = document.getElementById('llmDocType');
    if (docTypeEl) {
        docTypeEl.textContent = result.document_type_name || result.document_type || 'Unknown';
    }
}

// =============================================================================
// SCHEMA MANAGEMENT
// =============================================================================

function initSchemaManagement() {
    // Main tab switching
    const viewerTabBtn = document.getElementById('viewerTabBtn');
    const llmTabBtn = document.getElementById('llmTabBtn');
    const schemasTabBtn = document.getElementById('schemasTabBtn');
    
    if (viewerTabBtn) viewerTabBtn.addEventListener('click', () => switchMainTab('viewer'));
    if (llmTabBtn) llmTabBtn.addEventListener('click', () => switchMainTab('llm'));
    if (schemasTabBtn) schemasTabBtn.addEventListener('click', () => switchMainTab('schemas'));
    
    // LLM controls in main tab
    const pullModelBtnMain = document.getElementById('pullModelBtnMain');
    const processLlmBtnMain = document.getElementById('processLlmBtnMain');
    const llmModelSelectMain = document.getElementById('llmModelSelectMain');
    
    if (pullModelBtnMain) pullModelBtnMain.addEventListener('click', pullSelectedModelMain);
    if (processLlmBtnMain) processLlmBtnMain.addEventListener('click', processWithLlmMain);
    if (llmModelSelectMain) llmModelSelectMain.addEventListener('change', handleModelSelectMain);
    
    // Schema management buttons
    const newSchemaBtn = document.getElementById('newSchemaBtn');
    const saveSchemaBtn = document.getElementById('saveSchemaBtn');
    const deleteSchemaBtn = document.getElementById('deleteSchemaBtn');
    const generateSchemaBtn = document.getElementById('generateSchemaBtn');
    
    if (newSchemaBtn) newSchemaBtn.addEventListener('click', createNewSchema);
    if (saveSchemaBtn) saveSchemaBtn.addEventListener('click', saveCurrentSchema);
    if (deleteSchemaBtn) deleteSchemaBtn.addEventListener('click', deleteCurrentSchema);
    if (generateSchemaBtn) generateSchemaBtn.addEventListener('click', generateSchemaWithLLM);
    
    // YAML editor change tracking
    const yamlEditor = document.getElementById('schemaYamlEditor');
    if (yamlEditor) {
        yamlEditor.addEventListener('input', () => {
            state.schemaYamlDirty = true;
            document.getElementById('saveSchemaBtn').disabled = false;
        });
    }
}

function switchMainTab(tabName) {
    const viewerTabBtn = document.getElementById('viewerTabBtn');
    const llmTabBtn = document.getElementById('llmTabBtn');
    const schemasTabBtn = document.getElementById('schemasTabBtn');
    const viewerTab = document.getElementById('viewerTab');
    const llmMainTab = document.getElementById('llmMainTab');
    const schemasTab = document.getElementById('schemasTab');
    
    // Deactivate all tabs
    [viewerTabBtn, llmTabBtn, schemasTabBtn].forEach(btn => btn?.classList.remove('active'));
    [viewerTab, llmMainTab, schemasTab].forEach(tab => {
        if (tab) {
            tab.style.display = 'none';
            tab.classList.remove('active');
        }
    });
    
    if (tabName === 'viewer') {
        viewerTabBtn.classList.add('active');
        viewerTab.style.display = 'flex';
        viewerTab.classList.add('active');
    } else if (tabName === 'llm') {
        llmTabBtn.classList.add('active');
        llmMainTab.style.display = 'flex';
        llmMainTab.classList.add('active');
        // Load LLM status when switching to tab
        updateLlmStatusUIMain();
    } else if (tabName === 'schemas') {
        schemasTabBtn.classList.add('active');
        schemasTab.style.display = 'flex';
        schemasTab.classList.add('active');
        
        // Load schemas when switching to tab
        loadSchemaList();
        loadSchemaLlmModels();
    }
}

// LLM Main Tab Functions
function updateLlmStatusUIMain() {
    // Update status indicators in main LLM tab
    const ollamaStatus = document.getElementById('ollamaStatusMain');
    const ollamaStatusText = document.getElementById('ollamaStatusTextMain');
    const accelIcon = document.getElementById('accelIconMain');
    const accelText = document.getElementById('accelTextMain');
    const modelSelect = document.getElementById('llmModelSelectMain');
    const pullBtn = document.getElementById('pullModelBtnMain');
    const processBtn = document.getElementById('processLlmBtnMain');
    
    if (!ollamaStatus) return;
    
    if (state.llmStatus.ollamaAvailable) {
        ollamaStatus.textContent = '🟢';
        ollamaStatusText.textContent = 'Ollama Online';
        
        // Update acceleration info
        if (state.llmStatus.acceleration) {
            const accel = state.llmStatus.acceleration;
            if (accel === 'metal' || accel === 'mps') {
                accelIcon.textContent = '🍎';
                accelText.textContent = 'Apple Metal GPU';
            } else if (accel === 'cuda') {
                accelIcon.textContent = '🟢';
                accelText.textContent = 'NVIDIA CUDA';
            } else if (accel === 'rocm') {
                accelIcon.textContent = '🔴';
                accelText.textContent = 'AMD ROCm';
            } else {
                accelIcon.textContent = '💻';
                accelText.textContent = 'CPU Only';
            }
        }
        
        // Populate model select
        modelSelect.innerHTML = '';
        state.llmStatus.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name + (model.available ? ' ✓' : ' (not pulled)');
            option.dataset.available = model.available;
            modelSelect.appendChild(option);
        });
        modelSelect.disabled = false;
        
        // Select first available model
        const firstAvailable = state.llmStatus.models.find(m => m.available);
        if (firstAvailable) {
            modelSelect.value = firstAvailable.id;
            state.llmStatus.selectedModel = firstAvailable.id;
        }
        
        handleModelSelectMain();
    } else {
        ollamaStatus.textContent = '🔴';
        ollamaStatusText.textContent = 'Ollama Offline';
        accelIcon.textContent = '⚪';
        accelText.textContent = 'Not available';
        modelSelect.innerHTML = '<option value="">Ollama offline</option>';
        modelSelect.disabled = true;
        pullBtn.disabled = true;
        processBtn.disabled = true;
    }
}

function handleModelSelectMain() {
    const modelSelect = document.getElementById('llmModelSelectMain');
    const pullBtn = document.getElementById('pullModelBtnMain');
    const processBtn = document.getElementById('processLlmBtnMain');
    
    if (!modelSelect) return;
    
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    const isAvailable = selectedOption?.dataset.available === 'true';
    
    state.llmStatus.selectedModel = modelSelect.value;
    
    pullBtn.disabled = isAvailable;
    processBtn.disabled = !isAvailable || !state.ocrResult;
}

async function pullSelectedModelMain() {
    const modelSelect = document.getElementById('llmModelSelectMain');
    const pullBtn = document.getElementById('pullModelBtnMain');
    const progressContainer = document.getElementById('pullProgressMain');
    const progressFill = document.getElementById('pullProgressFillMain');
    const progressText = document.getElementById('pullProgressTextMain');
    
    const modelId = modelSelect.value;
    if (!modelId) return;
    
    pullBtn.disabled = true;
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting download...';
    
    try {
        const response = await fetch(`/api/llm/pull/${encodeURIComponent(modelId)}/stream`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.status === 'already_available') {
                            progressText.textContent = 'Model already available!';
                            setTimeout(() => { progressContainer.style.display = 'none'; }, 2000);
                            break;
                        }
                        
                        if (data.total && data.completed) {
                            const percent = Math.round((data.completed / data.total) * 100);
                            progressFill.style.width = percent + '%';
                            const completedMB = (data.completed / 1024 / 1024).toFixed(1);
                            const totalMB = (data.total / 1024 / 1024).toFixed(1);
                            progressText.textContent = `${completedMB} MB / ${totalMB} MB (${percent}%)`;
                        } else if (data.status) {
                            progressText.textContent = data.status;
                        }
                        
                        if (data.status === 'complete' || data.status === 'success') {
                            progressText.textContent = 'Download complete!';
                            progressFill.style.width = '100%';
                            await checkLlmStatus();
                            updateLlmStatusUIMain();
                            setTimeout(() => { progressContainer.style.display = 'none'; }, 2000);
                        }
                        
                        if (data.error) {
                            progressText.textContent = 'Error: ' + data.error;
                        }
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        console.error('Pull failed:', error);
        progressText.textContent = 'Pull failed: ' + error.message;
    }
    
    pullBtn.disabled = false;
}

async function processWithLlmMain() {
    if (!state.ocrResult) {
        alert('Please run OCR first');
        return;
    }
    
    const modelSelect = document.getElementById('llmModelSelectMain');
    const processBtn = document.getElementById('processLlmBtnMain');
    const placeholder = document.getElementById('llmPlaceholderMain');
    const resultDiv = document.getElementById('llmResultMain');
    const loadingDiv = document.getElementById('llmLoadingMain');
    
    const modelId = modelSelect.value;
    
    processBtn.disabled = true;
    placeholder.style.display = 'none';
    resultDiv.style.display = 'none';
    loadingDiv.style.display = 'flex';
    
    try {
        const textLines = state.ocrResult.text_lines || [];
        const fullText = textLines.map(line => line.text).join('\n');
        
        const documentType = state.ocrResult.document_class?.type_id || 
                            state.ocrResult.document_class?.class?.toLowerCase() || 
                            'unknown';
        
        const response = await fetch('/api/llm/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: fullText,
                model: modelId,
                document_type: documentType
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }
        
        const result = await response.json();
        state.llmResult = result;
        
        displayLlmResultMain(result);
        
    } catch (error) {
        console.error('LLM processing failed:', error);
        alert('LLM processing failed: ' + error.message);
        placeholder.style.display = 'flex';
    } finally {
        loadingDiv.style.display = 'none';
        processBtn.disabled = false;
    }
}

function displayLlmResultMain(result) {
    const placeholder = document.getElementById('llmPlaceholderMain');
    const resultDiv = document.getElementById('llmResultMain');
    const extractedFieldsDiv = document.getElementById('llmExtractedFieldsMain');
    const enhancedTextDiv = document.getElementById('llmEnhancedTextMain');
    const docTypeBadge = document.getElementById('llmDocTypeBadgeMain');
    const confidenceSection = document.getElementById('llmConfidenceSectionMain');
    const confidenceNotes = document.getElementById('llmConfidenceNotesMain');
    const modelUsed = document.getElementById('llmModelUsedMain');
    const docType = document.getElementById('llmDocTypeMain');
    
    placeholder.style.display = 'none';
    resultDiv.style.display = 'flex';
    
    // Display document type badge
    if (docTypeBadge) {
        docTypeBadge.textContent = result.document_type_name || result.document_type || '';
    }
    
    // Display extracted fields
    extractedFieldsDiv.innerHTML = '';
    if (result.extracted_fields) {
        for (const [key, value] of Object.entries(result.extracted_fields)) {
            const fieldDiv = document.createElement('div');
            fieldDiv.className = 'field-item';
            
            if (Array.isArray(value)) {
                fieldDiv.innerHTML = `<strong>${key}:</strong>`;
                const listDiv = document.createElement('div');
                listDiv.className = 'field-list';
                value.forEach((item, idx) => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'field-list-item';
                    if (typeof item === 'object') {
                        itemDiv.innerHTML = Object.entries(item)
                            .map(([k, v]) => `<span><em>${k}:</em> ${v || '-'}</span>`)
                            .join(' ');
                    } else {
                        itemDiv.textContent = item;
                    }
                    listDiv.appendChild(itemDiv);
                });
                fieldDiv.appendChild(listDiv);
            } else {
                fieldDiv.innerHTML = `<strong>${key}:</strong> ${value || '-'}`;
            }
            
            extractedFieldsDiv.appendChild(fieldDiv);
        }
    }
    
    // Display corrected text
    enhancedTextDiv.textContent = result.corrected_text || result.original_text || '';
    
    // Display confidence notes
    if (result.confidence_notes) {
        confidenceSection.style.display = 'block';
        confidenceNotes.textContent = result.confidence_notes;
    } else {
        confidenceSection.style.display = 'none';
    }
    
    // Display meta info
    modelUsed.textContent = result.model || 'Unknown';
    docType.textContent = result.document_type_name || result.document_type || 'Unknown';
}

async function loadSchemaList() {
    try {
        const response = await fetch('/api/schemas');
        if (!response.ok) throw new Error('Failed to load schemas');
        
        const data = await response.json();
        state.schemas = data.schemas;
        
        renderSchemaList();
    } catch (error) {
        console.error('Failed to load schemas:', error);
    }
}

function renderSchemaList() {
    const schemaList = document.getElementById('schemaList');
    if (!schemaList) return;
    
    schemaList.innerHTML = '';
    
    state.schemas.forEach(schema => {
        const item = document.createElement('div');
        item.className = 'schema-item' + (schema.type_id === state.selectedSchemaId ? ' active' : '');
        item.innerHTML = `
            <div class="schema-item-name">${schema.display_name}</div>
            <div class="schema-item-meta">${schema.field_count} fields • ${schema.type_id}</div>
        `;
        item.addEventListener('click', () => selectSchema(schema.type_id));
        schemaList.appendChild(item);
    });
}

async function selectSchema(typeId) {
    if (state.schemaYamlDirty) {
        if (!confirm('You have unsaved changes. Discard them?')) {
            return;
        }
    }
    
    state.selectedSchemaId = typeId;
    state.schemaYamlDirty = false;
    
    // Update list selection
    renderSchemaList();
    
    // Load schema YAML
    try {
        const response = await fetch(`/api/schemas/${typeId}/yaml`);
        if (!response.ok) throw new Error('Failed to load schema');
        
        const data = await response.json();
        
        // Show editor
        document.getElementById('schemaPlaceholder').style.display = 'none';
        document.getElementById('schemaEditForm').style.display = 'flex';
        document.getElementById('schemaEditorTitle').textContent = data.filename;
        document.getElementById('schemaYamlEditor').value = data.yaml;
        document.getElementById('saveSchemaBtn').disabled = true;
        document.getElementById('deleteSchemaBtn').disabled = typeId === 'unknown';
        
    } catch (error) {
        console.error('Failed to load schema:', error);
        alert('Failed to load schema: ' + error.message);
    }
}

function createNewSchema() {
    if (state.schemaYamlDirty) {
        if (!confirm('You have unsaved changes. Discard them?')) {
            return;
        }
    }
    
    state.selectedSchemaId = null;
    state.schemaYamlDirty = false;
    
    // Clear list selection
    renderSchemaList();
    
    // Show editor with template
    const template = `# New Document Schema
type_id: new_document
display_name: New Document Type

clip_prompts:
  - "a document"

keywords:
  - document

llm_context: |
  This is a NEW DOCUMENT TYPE. Describe what to extract here.

fields:
  - name: example_field
    type: text
    description: "Example field description"
    required: true
`;
    
    document.getElementById('schemaPlaceholder').style.display = 'none';
    document.getElementById('schemaEditForm').style.display = 'flex';
    document.getElementById('schemaEditorTitle').textContent = 'New Template';
    document.getElementById('schemaYamlEditor').value = template;
    document.getElementById('saveSchemaBtn').disabled = false;
    document.getElementById('deleteSchemaBtn').disabled = true;
}

async function saveCurrentSchema() {
    const yamlContent = document.getElementById('schemaYamlEditor').value;
    
    if (!yamlContent.trim()) {
        alert('Schema content is empty');
        return;
    }
    
    // Extract type_id from YAML
    const typeIdMatch = yamlContent.match(/type_id:\s*(\S+)/);
    const typeId = typeIdMatch ? typeIdMatch[1] : 'new_schema';
    
    try {
        const response = await fetch(`/api/schemas/${typeId}/yaml`, {
            method: 'PUT',
            headers: { 'Content-Type': 'text/plain' },
            body: yamlContent
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to save schema');
        }
        
        const data = await response.json();
        console.log('Schema saved:', data);
        
        state.schemaYamlDirty = false;
        state.selectedSchemaId = data.type_id;
        document.getElementById('saveSchemaBtn').disabled = true;
        
        // Reload schema list
        await loadSchemaList();
        
        alert('Schema saved successfully!');
        
    } catch (error) {
        console.error('Failed to save schema:', error);
        alert('Failed to save schema: ' + error.message);
    }
}

async function deleteCurrentSchema() {
    if (!state.selectedSchemaId || state.selectedSchemaId === 'unknown') {
        return;
    }
    
    if (!confirm(`Are you sure you want to delete the "${state.selectedSchemaId}" schema?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/schemas/${state.selectedSchemaId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete schema');
        }
        
        console.log('Schema deleted:', state.selectedSchemaId);
        
        state.selectedSchemaId = null;
        state.schemaYamlDirty = false;
        
        // Reset editor
        document.getElementById('schemaPlaceholder').style.display = 'flex';
        document.getElementById('schemaEditForm').style.display = 'none';
        document.getElementById('schemaEditorTitle').textContent = 'Select a template';
        
        // Reload schema list
        await loadSchemaList();
        
    } catch (error) {
        console.error('Failed to delete schema:', error);
        alert('Failed to delete schema: ' + error.message);
    }
}

async function loadSchemaLlmModels() {
    const modelSelect = document.getElementById('schemaLlmModel');
    if (!modelSelect) return;
    
    try {
        const response = await fetch('/api/llm/status');
        if (!response.ok) throw new Error('Failed to get LLM status');
        
        const data = await response.json();
        
        modelSelect.innerHTML = '';
        
        if (!data.ollama_available) {
            modelSelect.innerHTML = '<option value="">Ollama offline</option>';
            document.getElementById('generateSchemaBtn').disabled = true;
            return;
        }
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name + (model.available ? '' : ' (not pulled)');
            option.disabled = !model.available;
            modelSelect.appendChild(option);
        });
        
        // Select first available model
        const firstAvailable = data.models.find(m => m.available);
        if (firstAvailable) {
            modelSelect.value = firstAvailable.id;
            document.getElementById('generateSchemaBtn').disabled = false;
        } else {
            document.getElementById('generateSchemaBtn').disabled = true;
        }
        
    } catch (error) {
        console.error('Failed to load LLM models:', error);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

async function generateSchemaWithLLM() {
    const description = document.getElementById('schemaDescription').value.trim();
    if (!description) {
        alert('Please describe the document type you want to create');
        return;
    }
    
    const modelId = document.getElementById('schemaLlmModel').value;
    const generateBtn = document.getElementById('generateSchemaBtn');
    const yamlEditor = document.getElementById('schemaYamlEditor');
    
    generateBtn.disabled = true;
    generateBtn.textContent = '⏳ Generating...';
    
    // Show editor and clear it for streaming output
    document.getElementById('schemaPlaceholder').style.display = 'none';
    document.getElementById('schemaEditForm').style.display = 'flex';
    document.getElementById('schemaEditorTitle').textContent = 'Generating...';
    yamlEditor.value = '';
    
    try {
        const response = await fetch('/api/schemas/generate/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: description,
                model: modelId || null
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        if (data.token) {
                            // Append token to editor (streaming)
                            yamlEditor.value += data.token;
                            // Auto-scroll to bottom
                            yamlEditor.scrollTop = yamlEditor.scrollHeight;
                        }
                        
                        if (data.done) {
                            // Final result - update with cleaned YAML
                            if (data.yaml) {
                                yamlEditor.value = data.yaml;
                            }
                            document.getElementById('schemaEditorTitle').textContent = 
                                (data.display_name || 'New Schema') + ' (generated)';
                            
                            if (data.error) {
                                console.warn('Schema generation warning:', data.error);
                            }
                            
                            console.log('Schema generated:', data.type_id);
                        }
                    } catch (e) {
                        if (e.message !== 'Unexpected end of JSON input') {
                            console.error('Parse error:', e);
                        }
                    }
                }
            }
        }
        
        state.schemaYamlDirty = true;
        state.selectedSchemaId = null;
        document.getElementById('saveSchemaBtn').disabled = false;
        document.getElementById('deleteSchemaBtn').disabled = true;
        
    } catch (error) {
        console.error('Failed to generate schema:', error);
        alert('Failed to generate schema: ' + error.message);
        yamlEditor.value = '# Generation failed: ' + error.message;
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = '🤖 Generate';
    }
}
