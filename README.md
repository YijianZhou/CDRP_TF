# CDRP_TF
CNN Event Detection &amp; RNN Phase Picking (in Tensorflow)  
  
*tflib* defines genral methods in tensorflow  
> tflib
>> layers  
>> nn_model
  
*seisnet* defines neural networks for seismic data processing  
> seisnet
>> config  
>> data_pipeline  
>> models  
  
*preprocess* contains scripts for making TFRecords files  
> preprocess
>> make_all  
>> mk_dataset
  
run *train.py* to train the model  
*picker.py* connect DetNet and PpkNet as a hybrid architecture for extracting seismic arrivals from continuous waveforms.
```python
# use CDRP model
import picker
picker = picker.CDRP_Picker(out_file, ckpt_dir)
picker.pick(stream) # input obspy.stream
```

Please reference:
https://pubs.geoscienceworld.org/ssa/srl/article/90/3/1079/569837/hybrid-event-detection-and-phase-picking-algorithm
