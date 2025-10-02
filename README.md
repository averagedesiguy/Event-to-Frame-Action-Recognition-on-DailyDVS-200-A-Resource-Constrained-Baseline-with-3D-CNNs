# Event-to-Frame-Action-Recognition-on-DailyDVS-200-A-Resource-Constrained-Baseline-with-3D-CNNs
Event-to-Frame Action Recognition on DailyDVS-200: A Resource-Constrained Baseline with 3D CNNs
FACULTY OF ENGINEERING, COMPUTING AND THE ENVIRONMENT

School of Computer Science and Mathematics Kingston University London



MSc Data Science
Ansar Ahmad 
K2403224
Event-to-Frame Action Recognition on DailyDVS-200: A Resource-Constrained Baseline with 3D CNNs





Date: 01/10/2025
Supervisor: Professor Dimitrios Makris




WARRANTY STATEMENT
This is a student project. Therefore, neither the student nor Kingston University makes any warranty, express or implied, as to the accuracy of the data or conclusion of the work performed on the project and will not be held responsible for any consequences arising out of any inaccuracies or omissions therein.
 
Table of Contents
I.	Introduction	4
II.	Literature Review	5
III.	Dataset	6
IV.	Methodology	6
V.	Model Architecture and Training	7
VI.	Experiments and Results	8
VII.	Runtime and Resources Profiling	11
VIII.	Discussion	11
IX.	Limitations	12
Recommendations:	12
X.	Ethical and Deployment Considerations	12
XI.	Conclusion	12
XII.	References	14
XIII.	Appendix 1	14


 
Table of Figures
Figure 1:Subject Yawning	5
Figure 2: Workflow diagram	5
Figure 3: Model architecture	6
Figure 4: Train, Test and Validation division	8
Figure 5: Results Accuracy	8
Figure 6: Confusion Matrix (action 1- action 50)	9
Figure 7:Confusion Matrix (action 49- action 100)	9
Figure 8:Confusion Matrix (action 101- action 150)	9
Figure 9:Confusion Matrix (action 150 - action 200)	9

 
Event-to-Frame Action Recognition on DailyDVS-200: A Resource-Constrained Baseline with 3D CNNs
Ansar Ahmad
School of Computer Science and Mathematics Kingston University London, London, U.K K2403224@kingston.ac.uk
https://github.com/averagedesiguy/Event-to-Frame-Action-Recognition-on-DailyDVS-200-A-Resource-Constrained-Baseline-with-3D-CNNs


Abstract—  Event cameras are biologically-inspired sensors that capture asynchronously the changes in brightness and offer advantages like high dynamic range and motion blur-free where traditional cameras do not. In this paper, an end-to-end pipeline is reported on the DailyDVS 200 dataset, which is composed of 200 human activities.
Our two-stage process first converts native event streams into normal video frames, considering polarity and timestamp normalisation. Next, a 3D convolutional neural network (a pretrained TorchVision R3D-18) is learned on these videos using techniques such as class-balanced sampling, mixed-precision training, and label smoothing. To address computational constraints, we trained the model on four isolated class partitions and achieved 58–76% Top-1 accuracy depending on the partition.
The paper provides an in-depth description of trade-offs of converting event streams into frames, proposes typical implementation pitfalls, and gives directions toward improved event-native representations and stronger network backbones. The work is made reproducible by having full guides and templates.
Introduction
Most action recognition research has been driven by RGB video datasets such as Kinetics, UCF-101, and HMDB-51 (Kay et al., 2017)[6], where models learn motion and appearance features from full frames. Yet, conventional frame-based cameras are bound by inherent limitations: they record redundant data at predetermined frame rates, suffer from motion blur during high-speed motion, and saturate in high dynamic range scenes.
Conversely, event cameras generate data asynchronously, producing an event only when the brightness of a pixel changes. This creates a sparse spatiotemporal point cloud of events with microsecond latency and 120–140 dB dynamic range. These capabilities make event cameras highly appropriate for use in robotics, AR/VR, autonomous vehicles, and human-machine interfaces, particularly in challenging lighting environments or high-speed environments.
Even though event cameras have challenged low-level vision tasks like feature tracking, SLAM, and optical flow, there existed a serious dearth of a large-scale event-based dataset for human action recognition. The DailyDVS-200 dataset fills this gap by providing 200 classes of ordinary actions captured in the form of event streams. From an engineering practicality standpoint, most research groups already possess mature, frame-based video pipelines with pre-trained backbones. This motivates establishing a practical baseline by converting event sequences into frames for use with standard 3D CNNs to evaluate the effectiveness of this approach before exploring native event-based methods.
Using the DailyDVS-200 event streams, our goal is to build a reproducible, multi-class action recogniser under realistic resource constraints, such as Colab-class GPU memory and strict training budgets. Our research investigates whether a pre-trained 3D CNN—R3D-18—can achieve decent accuracy when fine-tuned from video representations of event data. We shall investigate under strict scrutiny which event-to-video transformation parameters—timestamp unit, polarity handling, frame rate, and spatial resolution—affect downstream learning most deeply. Meanwhile, we will explore how partitioning classes and balanced sampling strategies affect training stability and accuracy when resources are scarce. Finally, we will identify the leading failure modes in order to determine which among the improvements, i.e., conversion tweaks, alternative sampling schemes, or architecture adjustments, would likely yield maximum accuracy boosts.
Literature Review
Event cameras operate on a new paradigm where each pixel reports an event—a tuple of (x,y,t,p)—whenever the log-intensity change at the pixel surpasses a certain threshold. The sign, p∈{+1,−1}, is utilized to signal brightness increase (ON) or decrease (OFF). The sparse and asynchronous data streams provide microsecond latency and high dynamic range. Thorough analyses such as Gallego et al. (2020) [5] combine the physics of sensors, calibration methods, and algorithmic approaches and conclude that temporal coding is the inherent power of the technology.
For use as input to regular Convolutional Neural Networks (CNNs), the sparse event stream must be converted into a dense grid-based representation. Shared representations are voxel grids, in which the events are binned to a B×H×W tensor, where B represents the temporal bins; time surfaces, which capture the timestamp of the last event for each pixel with an exponential decay function; and the Event Spike Tensor (EST), which is learned jointly with the primary task network. These methods capture significantly more temporal and polarity information compared to naïve intensity frame accumulation.
Another strategy is to reconstruct standard video frames from event streams using models like E2VID (Rebecq et al., 2019)[8]. The technique is capable of generating high-frame-rate videos (hundreds or thousands of FPS), which is appealing for leveraging pre-trained, frame-based backbones and for human interpretability. But reconstructions introduce visual artefacts and are computationally expensive.
Early event-based datasets like the DVS Gesture Dataset (Amir et al., 2017)[1] primarily addressed action recognition as hand gesture classification, for which Spiking Neural Networks (SNNs) on neuromorphic hardware had significant latency and efficiency benefits. More recent work addresses full-body actions, merging state-of-the-art event representations with 3D CNNs or graph-based models. One such new trend is the use of hybrid pipelines that involve the integration of a native event stream (e.g., a voxel grid) with a reconstructed frame stream in order to capture both micro-temporal dynamics and coarse spatial context.
In traditional video analysis, 3D CNNs such as R(2+1)D (Tran et al., 2018)[12], R3D (Tran et al., 2015)[11], and I3D (Carreira & Zisserman, 2017)[3], and two-stream methods (Simonyan & Zisserman, 2014)[9], which pool RGB and optical flow together, have been dominant. The SlowFast network (Feichtenhofer et al., 2019)[4], with its double processing streams, demonstrated that high-rate temporal information is crucial for precise motion detection. Large-scale pre-training on datasets like Kinetics-400 (Kay et al., 2017)[6] remains a powerful approach, providing fast convergence and robust performance.
Whereas some event-based datasets for specific tasks are accessible, such as DHP19 for pose estimation and MVSEC or DSEC for autonomous driving, DailyDVS-200 (Barchid et al., 2024) [2] is notable. Its size (200 action classes) and scope (multi-location, actors, and backgrounds) make it well suited specifically to the task of testing the generalizability of event-based action recognition models.
Many mature deep learning solutions from video classification are also applicable perfectly for event-based train inputs. Techniques such as label smoothing (Szegedy et al., 2016)[10] for model calibration, warmup with cosine learning rate schedule (Loshchilov & Hutter, 2017)[7] for stabilizing initial training, and Exponential Moving Average of model weights to reduce generalization error are directly applicable. Balanced sampling also corrects class imbalance, and Test-Time Augmentation (TTA) reduces prediction variance.
Dataset
DailyDVS-200  is an ECCV-2024 benchmark for event-based action recognition over 200 typical action categories recorded in naturalistic environments and designed to give balanced, reproducible evaluation. It comprises over 22,000 event sequences recorded from 47 subjects, offering both subject diversity and scale, and the sequences contain 14 dense attributes (i.e., scene type, motion complexity, lighting, and indoor/outdoor), which enables controlled subgroup analyses. Data were captured at 320×240 resolution with a DVXplorer Lite event camera and provided in .aedat4 format; the event stream contains timestamp t, coordinates (x,y), and polarity p (positive 1, negative 0), making it simple to form frame, voxel, or EST-style representations. The sample can be seen in Figure 1 where the subject can be seen yawning. Official train/validation/test splits are provided and keyed to subject IDs to avoid identity leakage, and the repository [13] provides baselines for multiple models (e.g., C3D, I3D, SlowFast, Swin-T, Timesformer) to serve as reference performance. Data is downloadable through Google Drive and Baidu Netdisk, and the dataset is released under MIT licence with signed participant consent. The companion paper condenses the motivation, attributes, and benchmarks, positioning DailyDVS-200 as a strong foundation for future neuromorphic action-recognition research.
 
Figure 1:Subject Yawning

Methodology
We have three design principles. Firstly, practicality: we use video tools and pretrained weights in order to minimize engineering friction and keep prototyping fast. Secondly, reproducibility: all settings are made explicit and portable—notebooks ready for Colab, deterministic seeds, and saved dataset splits—in such a way that results may safely be rerun and compared. Thirdly, extensibility: all system pieces are designed as interchangeable components—the converter, dataset layer, model and trainer— which may safely be swapped or augmented without conflict, e.g. from simple event accumulation by voxelisation, or from swapping out the backbone while leaving the rest of the pipeline intact. The basic workflow can be seen in Figure 2.  The converter notebook performs deterministic conversion with the following steps
 
Figure 2: Workflow diagram
To resolve potential inconsistencies in the resolution of timestamps, which can be either microseconds or nanoseconds depending on recording or export software, the converter initially makes an educated guess at the suitable time unit. It does this by calculating the median time between consecutive events.
Upon this identification, each timestamp t is converted to seconds with the respective scale factor: t_s =t/s, where s is either 10e6 in the case of microseconds or 10e9 in the case of nanoseconds. This precise computation is essential to prevent temporal drift that would otherwise compromise the validity of the following frame binning process.
For a target frame rate F (e.g., 30 fps), we divide the time span into bins of width Δt = 1/F. For bin k, we accumulate signed events into an image buffer I_k(x, y):
I_k (x,y)=α∙∑_(i:t_i∈[k∆t,(k+1)∆t))▒〖p_i∙1[x_i=x,y_i=y],〗
where p_i ∈ {+1, −1} is polarity and α a scaling factor. We then zero centre and normalise I_k to [0, 255] (or [0, 1]) with optional clamping. Alternative variants keep two channels (ON/OFF) and combine later; our baseline fuses into one grayscale frame for simplicity and speed.
Frames are resized to 128×128. If the aspect ratio is other than the original one, we use letterboxing (padding with zero) to prevent distortion. Lastly, frames are saved as .mp4 with H.264 to make them compatible with ffmpeg and PyTorch video loaders.
Converter copies directory hierarchy from input (e.g., participant/action/sequence) and outputs consistent file names (e.g., .aedat4 is substituted by .mp4). A manifest CSV saves source path, output path, frames count, duration, and converting settings (FPS, resolution, polarity mode). This manifest is then processed by dataset loader.
In order to maintain Colab limitations but sample all 200 classes, we create four partitions of classes that all get trained with one and the same recipe. We apply stratified 70/15/15 split (train/validation/test) per partition for balancing per-split classes. No overlap across splits between sequences and save split indices to disk for reproducibility.
We sample once per-video a 64 frame clip with segmented temporal sampling: partition the frame index space into 64 segments and sample one frame from each with minimal jitter. This covers the full variability-laden video. At test time, we sample one deterministic centre clip; TTA calculates multiple clips (e.g., 3 or 5 spaced out) and averages over logits.
Spatial transformations. Training utilizes random resized crop (0.8–1.0 scale), horizontal flipping (p=0.5), and weak rotation (e.g., ±5°). Testing utilizes only resize + centre crop. We duplicate grayscale frame to 3 channels to align with Kinetics pretrained R3D 18 expectations.
We normalise frame intensities using dataset-wide statistics computed on the training data. Each (replicated) channel is z-scored with mean and standard deviation estimated on a representative training subset. If no such statistics are available, we apply a fixed normalisation with mean = 0.5 and standard deviation = 0.5. Because event-accumulated frames are counts over time rather than radiometric RGB intensities, their value distribution differs from natural images; empirical normalisation therefore improves optimisation stability and calibration during training and evaluation.
Model Architecture and Training
We take R3D 18 , a 3D extension of ResNet 18 using 3D kernels in the stem and residual blocks. Pretraining on Kinetics 400  provides good priors. We change out the final fully connected layer with a new head whose dimensionality is equal to that of the partition's number of classes.
Cross entropy loss with label smoothing (ε=0.1) is targeted. For addressing class imbalance we use WeightedRandomSampler on the training dataset such that uniform class sampling is estimated by each mini batch.
Training with AdamW (3e 4 learning rate, 0.02 weight decay), schedule with cosine decay and warmup (Loshchilov & Hutter, 2017)[7] (linear, ~5% steps), mixed precision (AMP), and clipping gradients (max norm 1.0). We keep track of EMA model weights (with decay ~0.999) and test EMA model on test and validation splits.
Other than label balancing and smoothing the classes, our two other regularisers for overfitting are EMA and data augmentation. We may employ early stopping over validation accuracy for very tight compute budgets, but we typically train for a specified number of epochs and keep the best validation checkpoint. 
We obtain Top 1 test set accuracy and macro avg precision/recall/F1. We display per-class metrics and confusion matrices (normalised over rows) to emphasise systematic confusions. For TTA we display absolute gain (in pp) over single clip evaluation and latency multiplier.
Conversion to frames imposes temporal quantisation at Δt=1/F and usually compresses polarity into intensity. Intricate motion cues (e.g., wrist flicks, brief gestures) will be blurred over bins. Yet, it facilitates plug and play transfer learning and significantly reduces engineering effort, which is desirable for baselines and ablation. Event native representations (voxel/EST/time surface) offer improved temporal fidelity at the expense of proprietary data pipes and, at times, memory usage.
 
Figure 3: Model architecture

Notebooks and their roles   
	Converter (Colab): mounts storage (Drive/local), discovers .aedat4 files, decodes events, accumulates frames at target FPS, resizes/letterboxes to 128×128, and writes .mp4 using OpenCV/ffmpeg. It produces a manifest CSV and logs per file summaries. Error handling covers missing paths, unreadable files, and codec fallbacks.

	DailyDVS_Improved3DConv… (four Colab notebooks): implement a shared training loop parameterized by (partition, class count, file roots). Components include dataset/loader, transform pipelines, model definition, optimiser + scheduler, EMA, checkpointing, evaluation, and report artefact generation (confusion matrices, classification reports, history JSONs, prediction CSVs)

Directory structure:
project_root/
  splits/
    A train.txt  A_val.txt  A_test.txt
    B train.txt  B_val.txt  B_test.txt
    ...
  experiments/
    ...
  notebooks/
    converter.ipynb
    DailyDVS_Improved3DConv1.ipynb
    DailyDVS_Improved3DConv2.ipynb
    DailyDVS_Improved3DConv3.ipynb
    DailyDVS_Improved3DConv4.ipynb


Every run provides a complete, verifiable history of training and evaluation. We save two checkpoints—best.pth, holding the best validation score's Exponential Moving Average (EMA) weights, and last.pth, final-epoch weights—to enable you to choose between optimal validation performance and final model state. A history file (JSON) dumps epoch-wise loss and accuracy on both train and val sets, learning rate schedule, and EMA metrics, making possible correct diagnoses and reproducible visualization. For test purposes, we print detailed reports: per-class precision, recall, and F1 metrics; confusion matrices both as images (for quick checking) and CSV (for later analysis); as well as a CSV of test-set prediction for error checking or ensembling. Config is taken verbatim in YAML/JSON, covering all hyper-params and run settings, with an idx2c.json file mapping indices to human-usable names—the latter crucial for interpreting metrics and probability output. Lastly, our output contains the conversion manifest, registering per-video metadata from the event-to-video pipeline, such that any result's origin (and by extension all preprocessing decisions from it) may be traced back from it.
Experiments and Results
We train on Google Colab GPU (A100) with mixed precision to save memory and make it possible to process larger batches. The effective batch size is adjusted to the GPU capability—the usual 4–8 clips per device—to best utilise it without running out of memory. We train every partition for 50 epochs with a cosine learning rate schedule with a 5%-size warm-up: the short warm-up stabilises early optimisation without overshoot, and the cosine decay promotes smooth convergence without_manual step drops. For evaluation purposes, we choose the checkpoint achieving best validation performance under EMA (Exponential Moving Average) of weights and test it on held-out split; test-time augmentation (optional) samples 3 clips per video to average prediction and reduce variance. We report Top-1 accuracy to measure general correctness, macro F1 to display behaviour under class imbalance, detailed per-class precision/recall/F1 to enable fine-grained diagnosis, and confusion matrices to visualise systematic confusions between categories. This configuration strikes a balance between practicality (fits into Colab), stability (warm-up + cosine + EMA), and rigorous evaluation (TTA + comprehensive metrics).
Complete training on the 200-class DailyDVS dataset was not feasible on free Google Colab due to hardware limitations (approximately 12–13 GB system RAM and ~16 GB GPU VRAM on an NVIDIA T4). End-to-end .aedat stream processing to voxel grids/event frames and normalization, together with multi-worker data loading and prefetching, placed original events and intermediate tensors in memory together, placing peak RAM well over budget. On the GPU, memory pressure again increased with a 200-way classification head: despite being smaller than convolutional maps, activations and gradients of the output layer increase with the number of classes, and—coupled with long temporal windows and non-trivial batch sizes—were sufficient to cause out-of-memory errors. In order to remain within resources, the dataset was divided into four mutually disjoint subsets of 50 classes and trained/evaluated separately. This reduced in-memory dataset footprint so fewer samples and intermediates queued at one time; cut classifier head and per-batch logits/gradients; and decreased epochs, thus limiting prefetch queue depth. These measures individually reduced peak RAM/VRAM below Colab's threshold and allowed all runs to complete successfully. Granting the trade-off, results are derived from four independent 50-way problems and thereby are not reporting confusions between classes assigned to independent splits.
Partition	Class	Total Videos	Train	Test	Validation
A	1-50	7165	5015	1075	1075
B	49-100	7132	4992	1070	1070
C	101-150	5395	3775	810	810
D	151-200	2591	1813	389	389
The approximate counts are:


Main results (single clip evaluation)
Figure 4: Train, Test and Validation division
Representative test set results (single clip) from our R3D 18 baseline are:
Partition	Top 1%	Macro F1
A	58-59	0.60
B	69	0.70
C	71-72	0.72
D	75-76	0.75
Figure 5: Results Accuracy
 
We note later partitions systematically outperform earlier ones, and various factors could cause this trend.
We trained and tested using a Kinetics-400–pretrained R3D-18 backbone on four 50-class subsets together spanning all 200 of the DailyDVS actions (to keep below Colab memory limits).
On the four runs, the test Top-1 accuracies were 0.586 (N=1,075), 0.691 (N=1,070), 0.715 (N=810), and 0.756 (N=389), with an average of micro-averaged Top-1 of 0.671 over N=3,344 test clips.
As a reference, single-model 200-class results are used in the standard DailyDVS-200 benchmark; typical Top-1 accuracies are C3D 21.99, I3D 32.30, R(2+1)D 36.06, SlowFast 41.49, TSM 40.87, Timesformer 44.25, and Swin-T 48.06; event-specific pipelines utilize EST 32.23, ESTF 24.68, GET 37.28, Spikformer 36.94, and SDT 35.43.
These figures mean that our split-wise outcomes are actually greater in size than the highest reported 200-way baseline, but are not equivalent since our testing breaks the task down into four separate 50-way sub-problems. This setup keeps the label space per trial down and therefore doesn't expose inter-subset confusions a true 200-class classifier would see, which has the effect of keeping accuracy down. Moreover, architectural and preprocessing choices differ: our pipeline uses a frame-based R3D-18 video backbone coupled with voxel/frame conversions from event streams, whereas the benchmark table averages frame-based and event-native strategies, more strictly limiting strict like-for-like comparison. In the thesis, we therefore present the four per-subset test accuracies with the micro-aggregated averaged one, and we cite the official 200-class baselines, noting explicitly the differences in evaluation protocol as the primary cause of the gap. 
One is intrinsic dataset difficulty: actions aggregated in later partitions possibly have more apparent, separable semantics or less inter-class overlap (e.g., fewer kinematically or visually similar categories), such that boundaries are simpler for the model to learn. Another plausibility is clip length: if later partitions include relatively longer segments, our segmented sampling pattern is likelier to encompass full (onset, core motion, resolution) action phases, thus enhancing temporal coverage and attenuating ambiguity at test. Note also that macro F1 closely trackingTop-1 accuracy indicates that class imbalance cannot spur results; WeightedRandomSampler is deemed to offer sufficiently balanced exposure at train time between frequent and rare classes, such that gains are spread rather than centred in head-classes. Though promising, these directions require checks for confounds such as distributional shifts in backgrounds, subjects, or conditions across partitions. Checks for similar label distributions, normalizing clip-duration effects, and ablating without segmented sampling would help ensure improved temporal capture (as opposed to incidental bias) drives observed uplift in performance
 
Figure 6: Confusion Matrix (action 1- action 50)
 
Figure 7:Confusion Matrix (action 49- action 100)
 
Figure 8:Confusion Matrix (action 101- action 150)
 
Figure 9:Confusion Matrix (action 150 - action 200)
Our confusion analysis exhibits systematic errors concentrated among kinematically similar motions (e.g., polishing vs. wiping and texting vs. typing), between mirror symmetries (left–right variants), and between different stages of the same activity (starts vs. endings).  A leading cause is the event-to-video accumulation step, which compresses fine spike time into coarse frames; this erases tempo, directionality, and micro-gesture markers that frequently disambiguate similar motion patterns. The model thus comes to rely on broad motion trajectories that resemble one another across classes and on sparse spatial structure with little appearance information, so the confusion matrices bunch into block-like aggregates: families of hand–surface interactions, device manipulations, or recurrent strokes sharing dynamics but not having distinguishing spatial markers. Mirror confusions are also built in by augmentation-induced invariances (e.g., horizontal flipping) and by lack of explicit left/right supervision. Phase errors result because early and late segments exhibit virtually identical local motion with reversed or attenuated tempo, which accumulation smears into a similar signature. Altogether, these causes explain why the classifier stumbles where fine temporal ordering and subtle kinematic details make the most difference.
Runtime and Resources Profiling
Throughput (clips per second) and GPU utilisation are significant for sizing and planning experiments. On a T4 Colab with mixed precision, an R3D-18 at 128×128 for 64 frames and batch size 8 can do on the order of a few dozen clips per second at inference; training is considerably slower owing to the extra compute and memory expense of backpropagation, data augmentation, and optimiser updates. Correct usage depends on keeping the GPU occupied: enable pinned-memory dataloaders with sufficient workers, prefetch batches, and minimise host–device transfers in the step loop. Memory scales roughly with B×T×H×W (and with channel widths of the model), where spatial dimensions H×W quadratically and temporal duration T linearly increase costs; training activations double this footprint further, so growth in any dimension necessitates trade-offs such as smaller batch sizes, gradient accumulation, activation checkpointing, or sequence length/resolution reduction. In practice, begin with setting the target temporal context, then adjust spatial resolution and batch size to achieve high, stable GPU utilisation (e.g., >85%) without out-of-memory errors. When pushing resolution or T, try mixed precision along with checkpointing to manage activation memory, and employ gradient accumulation to regain an effective larger batch while maintaining convergence behaviour.
Discussion
Event-derived videos are strong baselines: while accumulation destroys micro-temporal accuracy, the outcome frame sequences contain enough motion structure that a pre-trained 3D CNN is able to learn meaningful discriminants on a broad action taxonomy. Transfer learning is strongest in this case—R3D-18 pre-trained on Kinetics-400 (Kay et al., 2017)[6] adapts quickly to event-derived inputs, and only a simple replication of grayscale to three channels enables initial convolutional filters to behave as intended with minimal engineering. Fairness and stability rely on regularisation and sampling: label smoothing avoids overconfidence, EMA avoids optimisation noise and offers smoother validation curves, and a WeightedRandomSampler improves tail class exposure—achieved by macro-F1 tightly tracking Top-1 rather than lagging due to imbalance. 
Downstream, key conversion choices persist: unit timestamp errors (e.g., µs vs ms) quietly stretch or compress temporal dynamics; overly aggressive normalisation tends to erase informative contrast; and polarity handling—merging positive/negative events into one stream vs two channels—modifies class separability, with two-channel inputs being less likely to eliminate onset/offset cues that help differentiate analogous motions. In short, good results are not just due to the backbone but also from careful conversion hygiene and subtle regularisation that together leverage limited compute.
Limitations
Our current pipeline has several limitations that constrain both possible gains and error modes. We are losing information by binning events in time to make 30 fps frames and merging polarities: high-rate, short-duration spikes and fine-grained directional changes are smoothed out, eliminating cues such as micro-gestures and tempo; higher bin rates, adaptive voxelisation, or two-channel polarity inputs would maintain more discriminative timing. Second, partitioned training avoids memory use unpredictability through the ability to fit distinct heads per class block, but it also deprives the model of a global overview of inter-class boundaries; this can make cross-block ambiguities deeper, ones that a single, unified classifier—or periodic joint fine-tuning over blocks—could resolve. Third, while R3D-18 achieves a great speed–accuracy trade-off, it may underfit complex temporal relations; longer temporal receptive field models or multi-rate branches (e.g., SlowFast, temporal transformers, conformers) are better suited to identify long-range dependencies without relying exclusively on sequence length augmentation. Finally, there is a change in the domain: Pretraining on RGB images by Kinetics places appearance-driven priors that are less event-data-aligned; finetuning counteracts it, but there remains some capability that is left untapped. Event-aware adjustments—such as early fusion layers tuned to polarity and timestamp structuring, self-supervised pretraining on event streams, or light adapters inserted into pre-trained backbones—can bridge this gap while keeping the benefits of transfer learning.
Recommendations: 
Begin with cross-checking event-to-video conversion on a small test set: render and visually inspect a few clips to make sure polarity handling (e.g., positive/negative channels or combined polarity) looks reasonable, frame rate and time-binning (no µs↔ms mis-interpretings, no burst drops), and normalisation bounds; a quick "can the model overfit 10 clips?" smoke test identifies silent data problems early. Freeze the conversion manifest and train/val/test split indices as soon as the pipeline is healthy—modifying either mid-project is detrimental to comparability, introduces risks of data leakage, and renders ablations impossible. Start with a minimalist but evocative working point (T=64 frames, H=W=128): establish a constant case, define memory/throughput, then scale selectively based on error analysis—increase T when phase confusions are the error bottleneck, increase H/W if spatial detail matters, or increase polarity channels when direction is ambiguous—through gradient accumulation or activation checkpointing only where needed. 
Log everything for repeatability and auditability: configs and hyperparameters, random seeds, library and CUDA/cuDNN versions, exact dataset/split hashes, checkpoint digests, and the manifest snapshot.This rigor ensures that results are reproducible, differences are accountable, and next improvements are evidence-driven, not driven by drift.
Ethical and Deployment Considerations
Event data may be more privacy friendly than RGB, but actions may be sensitive. Obtain consent and offer secure storage, especially for human-focused applications. In deployment (e.g., robots, wearables), consider latency, power, and on-device inference; SNNs or quantized models may be warranted for power efficiency.
Conclusion
This report presented a reproducible complete baseline for action recognition on DailyDVS 200 through event→frame conversion and R3D 18 pretrained backbone. The pipeline, as simple as it is, achieves good mid to high two digit Top 1 across diverse actions. The recipe for engineering (cosine warm up, AMP, EMA, label smoothing, class balanced sampling) provides a good starting point for further research to build on.

In the future, the most viable strategy is to train one single model on all 200 classes in such a way that it will learn cross-partition regularities; class-balanced focal losses or curriculum learning can ease optimisation and stabilise imbalanced rare classes. Augmenting temporal and spatial context—e.g., 80–96 frames at 160–192 px—should enable more subtle motion cues, with mixed precision, gradient checkpointing, and (if needed) gradient accumulation remaining within Colab budgets. More robust backbones are an organic evolution: R(2+1)D-34, I3D-50, SlowFast-R50/101, or modern video transformers with frozen feature extractors augmented with lightweight adapters/LoRA to handle compute. Besides stacked frames, event-native representations (voxel grids, EST, time surfaces) deserve standalone use or two-stream fusion to preserve polarity and micro-timing; put it together with event-specific augmentations (e.g., EventDrop, polarity jitter, temporal dilations) and self-supervised pretraining on DailyDVS-200 (contrastive or masked-token objectives) to reduce label reliance. For classes whose discriminants are in ordering rather than posture, add fine-grained temporal supervision or explicit phase modelling (segment labels, temporal relation losses, CTC-style alignment). Finally, stress-test robustness to sensor noise, FPS variation, occlusions, and actor variance, and explore domain-generalisation strategies like style randomisation, strong augmentation/mixup, domain-adversarial objectives, or test-time adaptation. On balance, the conversion-plus-3D-CNN baseline is a sturdy, straightforward starting point which can be significantly improved.








 
References
	Amir, A., Taba, B., Berg, D., Melano, T., McKinstry, J., Di Nolfo, C., Nayak, T., Andreopoulos, A., Garreau, G., Mendoza, M. and DeBole, M., 2017. A low power, fully event-based gesture recognition system. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.7243–7252.
	Barchid, A., Binas, J., Aakur, S., Orchard, G., Liu, S., Delbruck, T. and Sironi, A., 2024. DailyDVS-200: A large-scale dataset for event-based human action recognition. European Conference on Computer Vision (ECCV).
	Carreira, J. and Zisserman, A., 2017. Quo vadis, action recognition? A new model and the Kinetics dataset. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.6299–6308.
	Feichtenhofer, C., Fan, H., Malik, J. and He, K., 2019. SlowFast networks for video recognition. Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp.6202–6211.
	Gallego, G., Delbruck, T., Orchard, G., Bartolozzi, C., Taba, B., Censi, A., Leutenegger, S., Davison, A.J., Conradt, J., Daniilidis, K. and Scaramuzza, D., 2020. Event-based vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(1), pp.154–180.
	Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P. and Suleyman, M., 2017. The Kinetics human action video dataset. arXiv preprint arXiv:1705.06950.
	Loshchilov, I. and Hutter, F., 2017. SGDR: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.
	Rebecq, H., Ranftl, R., Koltun, V. and Scaramuzza, D., 2019. Events-to-video: Bringing modern computer vision to event cameras. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.3857–3866.
	Simonyan, K. and Zisserman, A., 2014. Two-stream convolutional networks for action recognition in videos. Advances in Neural Information Processing Systems (NeurIPS), 27, pp.568–576.
	Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z., 2016. Rethinking the inception architecture for computer vision. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.2818–2826.
	Tran, D., Bourdev, L., Fergus, R., Torresani, L. and Paluri, M., 2015. Learning spatiotemporal features with 3D convolutional networks. Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp.4489–4497.
	Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y. and Paluri, M., 2018. A closer look at spatiotemporal convolutions for action recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.6450–6459.
	Wang, Q., Xu, Z., Lin, Y., Ye, J., Li, H., Zhu, G., Shah, S. A. A., Bennamoun, M. & Zhang, L. 2024, DailyDVS-200: A Comprehensive Benchmark Dataset for Event-Based Action Recognition, GitHub repository, viewed 1 October 2025, https://github.com/QiWang233/DailyDVS-200


Appendix 1
	"Be an expert in event-based vision. Describe my project (DailyDVS-200, event→frame conversion, R3D-18 backbone, Colab limits) in brief. Make a simple mental model diagram (ASCII fine) illustrating data flow from .aedat4 → frames → dataset → model → training → eval. Next, describe the most significant design decisions and trade-offs in bullet points.
Event→Video Converter (µs/ns detection)"
	"Write a Python script to receive DailyDVS .aedat4 event streams and output MP4 clips at 30 fps with the following features: microseconds vs nanoseconds detection (autodetection), polarity mode (1-channel and 2-channel modes), letterbox to 128×128, and manifest CSV (source path, output path, frames, duration, FPS, resolution, polarity mode). Following the code, walk through each function line by line. "
	Create a Colab-compatible PyTorch notebook to train an R3D-18 on the videos with augmentations: dataset class, transforms (train/test), WeightedRandomSampler, AMP mixed precision, AdamW, cosine scheduler with warmup, label smoothing, EMA weights, gradient clipping, checkpointing (best+last), and deterministic seeds. Every cell below, include a plain-English description of what it does.Dataset Partitions (A/B/C/D) + Splits
	"Code to split 200 classes into 4 disjoint sets of ~50 classes and then stratified 70/15/15 splits with index files saved. Add loaders to load and validate splits. Add assertions that train/val/test no sequence leakage holds. Go through each validation check in turn.Normalisation Stats + Dataloaders
	"Implement a utility that calculates dataset-level mean/std on train set (and use a representative sample if needed) and applies z-scoring of repeated grayscale channels. Implement DataLoader with pinned memory, workers, and prefetching. Describe performance trade-offs and demonstrate how to adjust num_workers and batch size on Colab T4 vs A100.
	"Add an evaluation module: Top-1 accuracy, macro precision/recall/F1, per-class report, and confusion matrix plots in PNG and CSV. Add a function to micro-average metrics over the four partitions. Following the code, indicate how to plot the confusion matrices with 2–3 examples of real-life instances.Test-Time Augmentation (TTA)"
	Add TTA for video classification: sample 3 or 5 clips per video at varied temporal offsets, average logits, and show absolute gain in percentage points over single-clip inference. Add a toggle flag and an estimate of latency multiplier. Note down the maths of logit averaging properly.
	Use utilities to capture VRAM usage, CPU RAM usage, and clips/sec throughput for both training and inference. Offer instructions on how to remain under Colab free tier (T4) constraints: gradient accumulation, activation checkpointing, reduced H×W/T, and batch size heuristics. Describe when to utilize each strategy.Ablation Harness
	“Create an ablation framework to sweep: FPS (15/30/60), polarity (merged vs 2-channel), temporal length (32/64/96), resolution (112/128/160), and optimiser/loss tweaks (weight decay, label smoothing ε). Save results to a CSV and plot accuracy vs setting. Provide a short template to write a one-paragraph ablation summary from the CSV.”
Explain Like I’m New to Event Vision
	Describe event cameras versus RGB frames in simple language, then apply that to how frame accumulation causes blurring of micro-temporal clues. Use brief metaphors and a tiny math example of binning events into a frame. Less than 400 words.
	"Insert modular code to transform input representation from stacked frames to voxel grids (B×H×W temporal bins). Optionally, insert a two-stream model (frames + voxel grid) with late fusion. Make it have a neat interface such that the training loop will not require touching. Describe architectural changes line by line.
	"Add pytest unit tests for: (a) µs/ns timestamp detection, (b) letterboxing correctness (aspect-ratio preserved), (c) manifest correctness, and (d) consistency of dataset indexing with stored splits. Include tiny synthetic event files in tests. Then describe how each test would detect a real bug.
	"Instrument the pipeline with structured logging (e.g., Python logging): conversion failure, unreadable files, codec fallbacks, and skipped samples. Include informative error messages and an end-of-run summary report. Include a 'dry-run' mode to check the data tree without performing work. Discuss how to debug using the logs.
	From my four partition outputs, please write a script to (a) calculate micro-averaged Top-1 on all test clips, (b) print a nicely formatted table (Markdown and CSV), and (c) automatically make a short comparative paragraph versus standard DailyDVS-200 baselines, explicitly declaring the non-equivalence of 50-class splits vs 200-class single-model. Provide the exact wording template. Colab Deployment Helper
	"Write one function make_colab_run() that mounts Google Drive, defines project paths, installs bare minimum dependencies, inspects GPU type, defines conservative defaults for T4 versus A100, and begins training on the selected partition. Post-code, provide a checklist to make everything functional in Colab."
	Line-by-Line Model Walkthrough "Print out the R3D-18 model overview (form, layers) and then go through step by step how a 64×128×128 clip is passed through the network. In simple terms, equate 3D convs to motion capture. Make it beginner-friendly but not inaccurate.
	Confusion Cluster Explorer
Apply code to automatically identify clusters of highly mixed-up classes within the confusion matrix (e.g., hierarchical clustering) and provide representative videos for every cluster. Then suggest two targeted cures for every cluster (e.g., raised T, polarity channels, particular augmentations). Record the clustering output.
