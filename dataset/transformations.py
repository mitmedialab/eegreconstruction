import numpy as np
import os
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def _generate_spectrogram(
        channel_data, 
        spectrogram_transform,
        sample_rate=250,
        f_min=1,
        f_max=95
        ):
    #channel_data = torch.Tensor(channel_data) #ensure data is tensor
    spectrogram = spectrogram_transform(channel_data) #calculate spectrogram
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram) #convert to DB
    freq_step = sample_rate / 2 / spectrogram.size(0) #calculate frequency step size

    # Find indices of frequency range
    idx_min = round(f_min / freq_step)
    idx_max = round(f_max / freq_step)

    spectrogram = spectrogram[idx_min:idx_max, :] #keep only frequencies in the desired range
    spectrogram = spectrogram.detach().numpy() #convert to numpy for image conversion & resizing
    spectrogram = np.interp(spectrogram, (spectrogram.min(), spectrogram.max()), (0, 255)) #covert to 0:255 range
    image = Image.fromarray(spectrogram.astype(np.uint8))
    image = image.resize((28, 224)) #resize to 224x28 to eventually combine all 8 into 224x224
    #Note: we resize from 97x16 to 224x28
    return image

def create_spectrogram_dataset(
        eeg_data,
        sample_rate=250, 
        n_fft=256, 
        win_length=256, 
        hop_length=256//8, 
        normalized=False,
        power=2.0,
        save_path = None
        ):
    """
    Generate a spectrogram image from EEG data for each sample in a .npy dataset.
    """
    eeg_data = torch.Tensor(eeg_data) #ensure data is tensor
    n_samples, n_timesteps, n_channels = eeg_data.shape #get shape of data
    spec_dataset = torch.empty(size=(n_samples, 3, 224, 224))#create container for spectrogram dataset

    #create spectrogram transform
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        normalized=normalized,
        power=power,
    )

    #image to tensor transform
    tensor_transform = transforms.PILToTensor()

    # Iterate over all samples in the EEG dataset
    for j in range(n_samples):
        combined_spectrogram = Image.new('L', (224, 224)) #create blank image to combine spectrograms into
        #Iterate over all channels in the EEG sample
        for i in range(n_channels):
            spectrogram_image = _generate_spectrogram(eeg_data[j,:, i], spectrogram_transform, sample_rate=sample_rate)
            combined_spectrogram.paste(spectrogram_image, (i*28, 0)) #paste channel spectrogram into combined spectrogram
        spec_dataset[j, :, :, :] = tensor_transform(combined_spectrogram).repeat(3,1,1) #copy along channel dimension and add combined spectrogram to dataset
    if save_path:
        np.save(save_path + "/" + "fourier_spectrograms.npy", spec_dataset)
    else:
        return spec_dataset

def create_grayscale_dataset(eeg_data, save_path = None):
    """
    Adapted from Mishra, A., Raj, N., & Bajwa, G. (2022). 
    EEG-based Image Feature Extraction for Visual Classification using Deep Learning. 
    DOI: 10.1109/IDSTA55301.2022.9923087
    
    This function converts the EEG signal to grayscale image
    
    parameters :
        eeg_data : eeg recording of shape (n_samples, n_timesteps, n_channels)

    returns :
        grayscale_image : array of 8 bit grayscale image stretched in 4 units
    """
    n_samples, n_timesteps, n_channels  = eeg_data.shape
    gray_dataset = torch.empty(size=(n_samples, 3, 224, 224)) #create container for grayscale dataset
    for i in range(n_samples):
        sample_output = torch.empty(size=(224, n_timesteps))
        for j in range(n_channels):
            x = torch.from_numpy(eeg_data[i, :, j])
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) #min-max scaling
            sample_output[j*28:(j+1)*28,:] = torch.tile(x, (28,1))
        sample_output = (sample_output*255).type(torch.uint8) #gray scale image with 8 bit (0-255) pixel values
        gray_dataset[i, :, :,:] = F.interpolate(sample_output[None, None, ...], size = (224,224)).repeat(1,3,1,1).squeeze() #resize to 3x224x224
        #Note: we interpolate from 224x500 to 224x224
    if save_path:
        np.save(save_path + "/" + "grayscale_images.npy", gray_dataset)
    else:
        return gray_dataset

def spectrogram_to_feature_vector(spectrograms, model_name = None, save_path = None):
    """
    Takes in a spectrogram and uses the pretrained model to extract a feature vector.
    Note: we only used EfficientNet but this function allows for other models, too.
    """
    # Load pretrained model
    if model_name.lower() == 'resnet50':
        model = models.resnet50(pretrained=True)
        out_dim = 2048
    elif model_name.lower() == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        out_dim = 1280
    elif model_name.lower() == 'regnet':
        model = models.regnet_y_3_2gf(pretrained=True)
        out_dim = 1512
    elif model_name.lower() == 'mobilenet':
        model = models.mobilenet_v3_large(pretrained=True)
        out_dim = 960
    else:
        raise ValueError('Model name not recognized. Please choose from resnet50, efficientnet, regnet, mobilenet.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model = model.to(device)
    model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the final layer
    model.eval() # Put model in evaluation mode

    # Put data on GPU
    spectrograms = spectrograms.to(device)

    # Take mean and std over one dataset (i.e. one mean & std for each channel)
    mean = spectrograms.reshape(-1,3,224*224).mean(axis=2).mean(axis=0)
    std = spectrograms.reshape(-1,3,224*224).std(axis=2).mean(axis=0)

    # Create feature vectors with pretrained model
    feature_vectors = np.empty(shape=(spectrograms.shape[0], out_dim)) # Create container for feature vectors
    for i, spectrogram in enumerate(spectrograms):
        spectrogram_norm = transforms.Normalize(mean, std)(spectrogram)
        with torch.no_grad():
            feature = model(spectrogram_norm.unsqueeze(0)) #spectrogram_norm
        feature_vectors[i,:] = feature.cpu().numpy().flatten()
    
    if save_path:
        np.save(save_path + "features_{}.npy".format(model_name), feature_vectors)
    else:
        return feature_vectors
    

def eeg_to_feature_vectors(
        dict_path = "./sven-thesis/data/preprocessed/wet/P001", 
        fourier = True,
        model_name = 'efficientnet'
        ):
    """
    Takes in a dictionary of EEG data and converts it to feature vectors.
    """
    # Load dictionary
    for run in os.listdir(dict_path):
        path = os.path.join(dict_path, run)
        eeg_data = np.load(os.path.join(path, "data.npy"))
    
        # Convert to spectrograms
        if fourier:
            spectrograms = create_spectrogram_dataset(eeg_data)
        else:
            spectrograms = create_grayscale_dataset(eeg_data)
        
        # Convert to feature vectors and save 
        spectrogram_to_feature_vector(spectrograms, model_name = model_name, save_path = os.path.join(path,'{}'.format("fourier_" if fourier else "grayscale_")))