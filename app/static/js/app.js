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
    deviceInfoExpanded: false
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
    clipDevice: document.getElementById('clipDevice')
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
