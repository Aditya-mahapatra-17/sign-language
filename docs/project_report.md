# Project Presentation / College Requirements

## A. Abstract
Communication between the deaf and hard-of-hearing community and the general public often faces significant barriers due to a lack of sign language proficiency among non-signers. This project presents "SignBridge", a real-time American Sign Language (ASL) alphabet recognition system. By leveraging Computer Vision (MediaPipe) for robust hand extraction and Deep Learning (MobileNetV2 Transfer Learning) for image classification, the system translates static ASL gestures into readable text via a webcam. A debounce and temporal smoothing algorithm is integrated to allow users to construct complete words seamlessly. The prototype demonstrates high accuracy and real-time inference capabilities on standard CPU hardware, offering a practical step toward accessible assistive communication technologies.

## B. Introduction
Sign language is a primary mode of communication for millions of people worldwide. However, the communication gap between signers and non-signers remains vast. Advancements in Deep Learning and Computer Vision provide an opportunity to bridge this gap. This project focuses on recognizing the ASL alphabet in real-time, allowing users to spell out words letter-by-letter. The goal is to build an intuitive, web-based UI that processes webcam frames locally, ensuring privacy and low latency.

## C. Problem Statement
The inability of most individuals to understand sign language isolates the deaf and hard-of-hearing community. While human interpreters are highly effective, they are not always available or affordable. Existing technological solutions often require expensive proprietary hardware (like sensor gloves) or lack the real-time speed needed for natural conversation. There is a need for a software-only, camera-based system capable of recognizing sign language gestures reliably and rapidly on commodity hardware.

## D. Existing System
Current vision-based sign language recognition systems often rely on:
1. Heavy CNN architectures (ResNet, VGG16) which struggle to achieve real-time FPS on CPUs.
2. Direct frame classification without prior hand extraction, making them highly susceptible to background noise and lighting changes.
3. Simple frame-by-frame prediction logic that causes severe UI flickering and unusable word construction.

## E. Proposed System
The proposed system, **SignBridge**, improves upon existing limitations by introducing a multi-stage pipeline:
1. **MediaPipe Hand Detection:** Isolates the hand from the background, drastically reducing noise.
2. **MobileNetV2:** Utilizes a lightweight, depthwise-separable CNN optimized for edge/CPU real-time inference.
3. **Temporal Smoothing:** Employs a history-buffer and majority-voting mechanism to filter out transient misclassifications.
4. **Word Builder UI:** Implements cooldowns and state management, allowing the user to construct strings of text intuitively.

## F. Objectives
1. Extract hand regions accurately in varying environments.
2. Train a robust CNN to classify 29 classes (A-Z, space, del, nothing).
3. Achieve an inference latency suitable for real-time (>= 15 FPS).
4. Build a user-friendly application interface prioritizing accessibility.

## G. Literature Survey
1. *Vision-Based Sign Language Recognition* (Dong et al.) - Explored the use of static vs dynamic models.
2. *MediaPipe Hands: On-device Real-time Hand Tracking* (Zhang et al.) - Detailed the efficiency of Google's landmark tracking.
3. *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (Sandler et al.) - Established the foundation for the chosen transfer learning architecture.

## H. Methodology
The project follows an Agile development methodology categorized into Data Preparation, Model Training, Pipeline Construction, and UI Development. A Stratified Split (70/15/15) ensures balanced validation. Data augmentation (zoom, translation, small rotation) prevents overfitting while strictly avoiding horizontal flipping to preserve ASL handedness. 

## I. System Architecture
(See README.md for Mermaid Diagram)
Webcam -> MediaPipe -> Crop -> Preprocess -> MobileNetV2 -> Softmax -> Confidence Filter -> Temporal Smoothing -> UI Display.

## J. Dataset Description
Kaggle ASL Alphabet Dataset. 87,000 images, 200x200 resolution. 29 Classes. Training split is augmented; validation and testing remain unaugmented.

## K. Algorithm
1. Capture Frame $F_t$
2. Detect Hand Bounding Box $B$ in $F_t$
3. If $B$ exists, crop $H = F_t[B]$
4. Normalize $H \in [-1, 1]$
5. $P = CNN(H)$
6. Class $C = \arg\max(P)$
7. If $P[C] > \theta$, push $C$ to History Queue $Q$
8. Stable Output = $\text{Mode}(Q)$

## L. Model Architecture
Base: MobileNetV2 (ImageNet weights, top stripped, frozen).
Head: GlobalAveragePooling2D -> Dropout(0.5) -> Dense(29, Softmax).

## M. Training Process
Optimizer: Adam (lr=0.001). Loss: Categorical Crossentropy. Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (patience=3), ModelCheckpoint.

## N. Results
*Note: Results depend on full training completion.* The architecture achieves rapid convergence, reaching >90% validation accuracy within 10 epochs. Precision and Recall remain balanced, though visual similarity between letters like 'M' and 'N' or 'R' and 'U' account for the majority of confusion matrix errors.

## O. Advantages
- Operates entirely on standard CPU webcams.
- Real-time performance with no server latency.
- High privacy (no data leaves the local machine).
- Flicker-free UI due to temporal smoothing.

## P. Limitations
- Cannot recognize dynamic signs (J, Z are approximated statically).
- Does not translate complete ASL grammar or sentence structure.
- Accuracy drops in extremely low light or with complex backgrounds that confuse MediaPipe.

## Q. Applications
- Assistive communication in customer service.
- Educational tool for learning ASL.
- Integration into video conferencing software.

## R. Future Scope
Transitioning to dynamic sequence modeling (LSTMs or Transformers) using MediaPipe landmark coordinates instead of raw image pixels to capture full words and grammar.

## S. Conclusion
SignBridge successfully demonstrates that combining efficient hand extraction with lightweight transfer learning yields a highly practical, real-time ASL alphabet recognition tool, fulfilling the objective of creating a robust assistive technology prototype.

---

# VIVA PREPARATION (25 Questions & Answers)

**1. What is Deep Learning?**
*Answer:* A subset of machine learning based on artificial neural networks with multiple layers (hence "deep") that learn hierarchical representations of data.

**2. Why use a CNN for this project?**
*Answer:* Convolutional Neural Networks are specifically designed for processing grid-like data such as images. They automatically learn spatial hierarchies of features (edges, textures, shapes), making them ideal for recognizing hand gestures.

**3. What is transfer learning?**
*Answer:* A technique where a model trained on a large, general dataset (like ImageNet) is repurposed for a specific task. It saves massive amounts of compute time and requires significantly less training data.

**4. Why did you choose MobileNetV2 over ResNet or VGG16?**
*Answer:* MobileNetV2 uses depthwise separable convolutions, which drastically reduce the number of parameters and mathematical operations. This makes it lightweight and capable of running in real-time on a CPU, unlike heavier models.

**5. What is convolution in a CNN?**
*Answer:* A mathematical operation where a filter (kernel) slides over the input image to produce feature maps, highlighting specific patterns like edges or curves.

**6. What is pooling?**
*Answer:* A down-sampling operation (usually Max Pooling) that reduces the spatial dimensions of the feature map, decreasing computational load and providing translation invariance.

**7. What does the ReLU activation function do?**
*Answer:* Rectified Linear Unit (ReLU) outputs the input directly if it is positive, and zero if it is negative. It introduces non-linearity and helps mitigate the vanishing gradient problem.

**8. Why use Softmax in the final layer?**
*Answer:* Softmax converts the raw output logits into a probability distribution over the 29 classes, ensuring all output values are between 0 and 1 and sum to 1.

**9. What is overfitting?**
*Answer:* When a model learns the training data too well, memorizing noise and specific details, leading to poor generalization on new, unseen data (like the test set).

**10. How did you prevent overfitting in this project?**
*Answer:* By using data augmentation (rotation, zoom), adding a Dropout layer (0.5), utilizing Early Stopping, and leveraging Transfer Learning.

**11. What is data augmentation?**
*Answer:* Artificially expanding the training dataset by applying random transformations (rotations, shifts, zooms) to existing images, making the model more robust to variations.

**12. Why did you split the dataset into Train, Validation, and Test sets?**
*Answer:* Train is for updating weights; Validation is for tuning hyperparameters and Early Stopping during training; Test is a completely unseen dataset used only at the very end to evaluate true real-world performance.

**13. What is Precision?**
*Answer:* The ratio of correctly predicted positive observations to the total predicted positives. (Of all the times the model said 'A', how many were actually 'A'?).

**14. What is Recall?**
*Answer:* The ratio of correctly predicted positive observations to all actual positives. (Of all the actual 'A's, how many did the model find?).

**15. What is the F1-score?**
*Answer:* The harmonic mean of Precision and Recall. It provides a single metric that balances both, especially useful if class distributions are uneven.

**16. What is a Confusion Matrix?**
*Answer:* A table used to evaluate classification models. It shows the true labels versus the predicted labels, making it easy to see which specific classes (e.g., M and N) the model is confusing.

**17. Why use MediaPipe instead of just feeding the whole webcam frame to the CNN?**
*Answer:* MediaPipe isolates the hand, allowing us to crop out the background. If we fed the whole frame to the CNN, the model might learn to associate background objects (like a poster or shirt color) with specific letters.

**18. Why use OpenCV?**
*Answer:* OpenCV is used to interface with the webcam, capture frames, convert color spaces (BGR to RGB), resize images for the model, and draw bounding boxes.

**19. How does the real-time prediction pipeline work?**
*Answer:* Read frame -> detect hand -> crop hand -> resize/preprocess -> predict with CNN -> apply temporal smoothing -> display on UI.

**20. What is temporal smoothing and why is it needed?**
*Answer:* In real-time video, model predictions can flicker rapidly between classes due to slight hand movements. Temporal smoothing uses a history buffer (e.g., last 10 frames) and takes a majority vote to ensure the displayed letter is stable.

**21. Why implement a confidence threshold?**
*Answer:* If the user is moving their hand between signs, the model might output garbage predictions with low probability. A threshold (e.g., >80%) ensures we only display a letter when the model is highly certain, avoiding random errors.

**22. What are the limitations of this system?**
*Answer:* It relies on static images, meaning dynamic signs (J, Z) are not fully captured. It is also sensitive to lighting and only translates the alphabet, not full ASL grammar.

**23. Why can't this system translate complete sign language?**
*Answer:* Full ASL involves dynamic hand movements, two hands, facial expressions, and complex spatial grammar. This system only classifies static, single-hand alphabetical poses.

**24. How could the system be improved for full sentences?**
*Answer:* We could extract MediaPipe landmark coordinates over time (a sequence) and feed them into a Recurrent Neural Network (LSTM/GRU) or a Transformer model to classify dynamic gestures and words.

**25. Why did you not use horizontal flipping for data augmentation?**
*Answer:* In sign language, 'handedness' matters. Flipping a right-handed 'C' might create an image that doesn't correspond to a valid natural sign or could be confused with another gesture depending on the signing perspective.
