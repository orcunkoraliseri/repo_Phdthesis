# MODEL PROPERTIES -----------------------------------------------------------------------------------------------------
# MODEL: CHECK--------------------------------------------------------------------------------------------------------
def check_activations_for_saturation(layer_outputs):
    sigmoid = torch.nn.Sigmoid()
    tanh = torch.nn.Tanh()

    for idx, output in enumerate(layer_outputs):
        sigmoid_out = sigmoid(output)
        tanh_out = tanh(output)

        sigmoid_saturation = ((sigmoid_out > 0.99) | (sigmoid_out < 0.01)).float().mean().item()
        tanh_saturation = ((tanh_out > 0.99) | (tanh_out < -0.99)).float().mean().item()

        print(f'Layer {idx} - Sigmoid saturation: {sigmoid_saturation:.4f}, Tanh saturation: {tanh_saturation:.4f}')

# RNNs:MODEL------------------------------------------------------------------------------------------------------------
import torch.nn as nn
class RNNsModel(nn.Module):
    def __init__(self,num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat, num_numFamMembCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers, hidden_units,
                 rnn_type='LSTM',
                 activation_func_act = "relu",
                 activation_func_bi="tanh",
                 dropout_loc = 0.5, dropout_withNOB = 0.5, dropout_embedding=0.5, dropout_RNNs=0.5,
                 embed_size=50):

        super(RNNsModel, self).__init__()
        embed_size = embed_size
        # Batch normalization layers are included to help stabilize and speed up the training process.
        self.rnn_type = rnn_type

        if activation_func_act == 'relu':
            self.activation_act = nn.ReLU()
        elif activation_func_act == 'tanh':
            self.activation_act = nn.Tanh()
        elif activation_func_act == 'sigmoid':
            self.activation_act = nn.Sigmoid()

        if activation_func_bi == 'relu':
            self.activation_binary = nn.ReLU()
        elif activation_func_bi == 'tanh':
            self.activation_binary = nn.Tanh()
        elif activation_func_bi == 'sigmoid':
            self.activation_binary = nn.Sigmoid()

        # Define embedding dimensions for all categorical features
        # Occupant Demographics
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 1)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 1)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 1)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        # Occupant Demographics
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the LSTM layers
        total_embedding_dim = sum([
            # Occupant Demographics
            self.embedding_dim_education,
            self.embedding_dim_employment,
            self.embedding_dim_gender,
            self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb,
            # Order columns
            self.embedding_dim_OCCinHH,
            # non-temporal TUS daily features
            self.embedding_dim_season,
            self.embedding_dim_weekend,
        ])
        self.input_size = total_embedding_dim + num_continuous_features
        #print("hello: input size is", input_size)

        # Dynamic LSTM layers initialization
        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.multiplier = 2 # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            if self.rnn_type == 'LSTM':
                rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            elif self.rnn_type == 'RNN':
                rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            elif self.rnn_type == 'GRU':
                rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            else:
                raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', 'RNN'.")
            self.shared_rnns.append(rnn_layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * self.multiplier))
            self.dropouts.append(nn.Dropout(p=dropout_RNNs))
            input_size = hidden_size * self.multiplier

        # Activity output layer
        self.activity_batch_norm = nn.BatchNorm1d(input_size)
        self.activity_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_activity)  # Assuming bidirectional LSTM

        # Location output layer with Dropout: Dropout is used to help prevent overfitting
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.location_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_location)  # Assuming bidirectional LSTM

        # withNOBODY output layer with Dropout: Dropout is used to help prevent overfitting
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)
        self.withNOB_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_withNOB)  # Assuming bidirectional LSTM
    def forward(self,  education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                OCCinHH_input,
                season_input, weekend_input,
                continuous_input):

        # Embeddings
        education_embedded = self.education_embedding(education_input).reshape(-1, 48, self.embedding_dim_education)
        #print("Initial Education Embedding: ", education_embedded)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 48,self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 48, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 48,self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 48,self.embedding_dim_numFamMemb)
        # Order columns
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 48, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((education_embedded,employment_embedded,gender_embedded, famTypology_embedded, numFamMemb_embedded,
                                           OCCinHH_embedded, season_embedded, weekend_embedded, continuous_input), dim=2)
        rnn_out = concatenated_features

        # Dynamic RNNs processing
        for rnn_layer, batch_norm, dropout in zip(self.shared_rnns, self.batch_norms, self.dropouts):
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = batch_norm(rnn_out.reshape(-1, rnn_out.shape[2])).reshape(rnn_out.size(0), -1, rnn_out.shape[2])
            rnn_out = dropout(rnn_out)  # Dropout applied here, controlled by dropout_RNNs parameter

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(self.activation_act(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        activity_output = activity_output.reshape(-1, 48, activity_output.shape[-1])  # Reshape to sequence form

        # Location with Dropout
        location_output = self.location_dropout(rnn_out)
        location_output = self.location_dense(self.activation_binary(location_output.contiguous().reshape(-1, location_output.shape[2])))
        location_output = location_output.reshape(-1, 48, location_output.shape[-1])  # Reshape to sequence form

        # WithNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(rnn_out)
        withNOB_output = self.withNOB_dense(self.activation_binary(withNOB_output.contiguous().reshape(-1, withNOB_output.shape[2])))
        withNOB_output = withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])  # Reshape to sequence form

        return activity_output, location_output, withNOB_output

# RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING
# RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING RNNs LESS-EMBEDDING
import torch.nn as nn
class RNNsModelLessEmbed(nn.Module):
    def __init__(self, num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity,
                 num_hidden_layers, hidden_units,
                 rnn_type='LSTM',
                 activation_func_act = 'relu',
                 dropout_embedding=0.5, dropout_RNNs=0.5,
                 embed_size=50,
                 normalization='batch'):

        super(RNNsModelLessEmbed, self).__init__()
        embed_size = embed_size
        # Batch normalization layers are included to help stabilize and speed up the training process.
        self.rnn_type = rnn_type
        self.normalization = normalization  # Store the normalization type

        # Activation functions
        self.activation_act = nn.ReLU() if activation_func_act == 'relu' else nn.Tanh()

        # Define embedding dimensions for all categorical features
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the LSTM layers
        total_embedding_dim = sum([
            # non-temporal TUS daily features
            self.embedding_dim_season,
            self.embedding_dim_weekend,
        ])
        input_size = total_embedding_dim + num_continuous_features
        #print("hello: input size is", input_size)

        # Dynamic LSTM layers initialization
        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        """
        self.batch_norms = nn.ModuleList()
        """
        self.norm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.multiplier = 2 # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            rnn_layer = self._create_rnn_layer(rnn_type, input_size, hidden_size)
            self.shared_rnns.append(rnn_layer)
            """
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * self.multiplier))
            """
            if self.normalization == 'batch':
                norm_layer = nn.BatchNorm1d(hidden_size * self.multiplier)
            elif self.normalization == 'layer':
                norm_layer = nn.LayerNorm(hidden_size * self.multiplier)
            else:
                raise ValueError("Invalid normalization type. Choose 'batch' or 'layer'.")
            self.norm_layers.append(norm_layer)

            self.dropouts.append(nn.Dropout(p=dropout_RNNs))
            input_size = hidden_size * self.multiplier

        # Activity output layer
        self.activity_batch_norm = nn.BatchNorm1d(input_size)
        self.activity_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_activity)  # Assuming bidirectional LSTM
    def _create_rnn_layer(self, rnn_type, input_size, hidden_size):
        if rnn_type == 'LSTM':
            return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        elif rnn_type == 'RNN':
            return nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        elif rnn_type == 'GRU':
            return nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', 'RNN'.")
    def forward(self, season_input, weekend_input, continuous_input):

        # Embeddings
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((season_embedded, weekend_embedded, continuous_input), dim=2)
        rnn_out = concatenated_features

        """
        # Dynamic RNNs processing
        for rnn_layer, batch_norm, dropout in zip(self.shared_rnns, self.batch_norms, self.dropouts):
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = batch_norm(rnn_out.reshape(-1, rnn_out.shape[2])).reshape(rnn_out.size(0), -1, rnn_out.shape[2])
            rnn_out = dropout(rnn_out)  # Dropout applied here, controlled by dropout_RNNs parameter
        """

        # Dynamic RNNs processing
        for rnn_layer, norm_layer, dropout in zip(self.shared_rnns, self.norm_layers, self.dropouts):
            rnn_out, _ = rnn_layer(rnn_out)
            if self.normalization == 'batch':
                rnn_out = norm_layer(rnn_out.reshape(-1, rnn_out.shape[2])).reshape(rnn_out.size(0), -1, rnn_out.shape[2])
            else:  # layer normalization
                rnn_out = norm_layer(rnn_out)
            rnn_out = dropout(rnn_out)

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(self.activation_act(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        activity_output = activity_output.reshape(-1, 48, activity_output.shape[-1])  # Reshape to sequence form

        return activity_output

# LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING
# LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING LSTM NO-EMBEDDING

# LSTM NO-EMBEDDING ----------------------------------------------------------------------------------------------------
import torch.nn as nn
class RNNsModelNoEmbed(nn.Module):
    def __init__(self, input_size, output_dim_activity, output_dim_location, output_dim_withNOB, num_hidden_layers, hidden_units,
                 rnn_type='LSTM', activation_func_act='relu', activation_func_bi = "tanh",
                 dropout_loc=0.5, dropout_withNOB=0.5, dropout_RNNs=0.5):
        super(RNNsModelNoEmbed, self).__init__()
        self.rnn_type = rnn_type

        if activation_func_act == 'relu':
            self.activation_act = nn.ReLU()
        elif activation_func_act == 'tanh':
            self.activation_act = nn.Tanh()
        elif activation_func_act == 'sigmoid':
            self.activation_act = nn.Sigmoid()
        elif activation_func_act == 'leakyRelu':
            self.activation_act = nn.LeakyReLU()

        if activation_func_bi == 'relu':
            self.activation_binary = nn.ReLU()
        elif activation_func_bi == 'tanh':
            self.activation_binary = nn.Tanh()
        elif activation_func_bi == 'sigmoid':
            self.activation_binary = nn.Sigmoid()
        elif activation_func_bi == 'leakyRelu':
            self.activation_binary = nn.LeakyReLU()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.multiplier = 2  # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            if self.rnn_type == 'LSTM':
                rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            elif self.rnn_type == 'GRU':
                rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            elif self.rnn_type == 'RNN':
                rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            else:
                raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', 'RNN'.")
            self.shared_rnns.append(rnn_layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * self.multiplier))
            self.dropouts.append(nn.Dropout(p=dropout_RNNs))
            input_size = hidden_size * self.multiplier

        # Activity output layer
        self.activity_batch_norm = nn.BatchNorm1d(input_size)
        self.activity_dense = nn.Linear(input_size, output_dim_activity)  # Assuming bidirectional RNN

        # Location output layer with Dropout: Dropout is used to help prevent overfitting
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.location_dense = nn.Linear(input_size, output_dim_location)  # Assuming bidirectional RNN

        # withNOBODY output layer with Dropout: Dropout is used to help prevent overfitting
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)  # Assuming bidirectional RNN

    def forward(self, continuous_input):

        rnn_out = continuous_input

        # Dynamic RNNs processing
        for rnn_layer, batch_norm, dropout in zip(self.shared_rnns, self.batch_norms, self.dropouts):
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = batch_norm(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])).reshape(-1, 48, rnn_out.shape[2])
            rnn_out = dropout(rnn_out)  # Dropout applied here, controlled by dropout_RNNs parameter

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(self.activation_act(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        activity_output = activity_output.reshape(-1, 48, activity_output.shape[-1])  # Reshape to sequence form

        # Location with Dropout
        location_output = self.location_dropout(rnn_out)
        location_output = self.location_dense(self.activation_binary(location_output.contiguous().reshape(-1, location_output.shape[2])))
        location_output = location_output.reshape(-1, 48, location_output.shape[-1])  # Reshape to sequence form

        # WithNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(rnn_out)
        withNOB_output = self.withNOB_dense(self.activation_binary(withNOB_output.contiguous().reshape(-1, withNOB_output.shape[2])))
        withNOB_output = withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])  # Reshape to sequence form

        return activity_output, location_output, withNOB_output

# LSTM NO-EMBEDDING: MODEL SIMPLER -------------------------------------------------------------------------------------
import torch.nn as nn
class RNNsModelNoEmbed_Simpler(nn.Module):
    def __init__(self, input_size, output_dim_activity, output_dim_location,num_hidden_layers,
                 hidden_units):
        super(RNNsModelNoEmbed_Simpler, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        self.activation = nn.ReLU()
        self.multiplier = 2  # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True,
                                bidirectional=True)
            self.shared_rnns.append(rnn_layer)
            input_size = hidden_size * self.multiplier

        # Output layers for activity, location, and withNOBODY
        self.activity_dense = nn.Linear(input_size, output_dim_activity)  # Assuming bidirectional RNN
        self.location_dense = nn.Linear(input_size, output_dim_location)  # Assuming bidirectional RNN

    def forward(self, continuous_input):
        rnn_out = continuous_input

        # Dynamic RNNs processing
        for rnn_layer in self.shared_rnns:
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = self.activation(rnn_out)

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY
        # Activity
        activity_output = self.activity_dense(self.activation(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        activity_output = activity_output.reshape(-1, 48, activity_output.shape[-1])  # Reshape to sequence form

        # Location
        location_output = self.location_dense(self.activation(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        location_output = location_output.reshape(-1, 48, location_output.shape[-1])  # Reshape to sequence form

        return activity_output, location_output
import torch
import torch.nn as nn
class RNNsModelNoEmbed_Simplest(nn.Module):
    def __init__(self, input_size, output_dim_location, num_hidden_layers, hidden_units):
        super(RNNsModelNoEmbed_Simplest, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # BatchNorm layers
        self.multiplier = 2  # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            self.shared_rnns.append(rnn_layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * self.multiplier))  # BatchNorm for each bidirectional output
            input_size = hidden_size * self.multiplier

        # Output layer for location
        self.location_dense = nn.Linear(input_size, output_dim_location)  # Assuming bidirectional RNN

    def forward(self, continuous_input):
        rnn_out = continuous_input

        # Dynamic RNNs processing
        for rnn_layer, batch_norm in zip(self.shared_rnns, self.batch_norms):
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = rnn_out.contiguous().reshape(-1, rnn_out.shape[2])  # Flatten batch for batch norm
            rnn_out = batch_norm(rnn_out)
            rnn_out = rnn_out.reshape(-1, 48, rnn_out.shape[1])  # Reshape back to sequence form

        # Location
        location_output = self.location_dense(rnn_out.contiguous().reshape(-1, rnn_out.shape[2]))
        location_output = location_output.reshape(-1, 48, location_output.shape[-1])  # Reshape to sequence form

        return location_output
# LSTM TUNING-----------------------------------------------------------------------------------------------------------
#_______________________________________________________________________________________________________________________
def weights_init_glorot_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
def weights_init_he_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)
def weights_init_lecun_normal(m):
    import math
    if isinstance(m, nn.Linear):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = 1.0 / math.sqrt(fan_in)
        torch.nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
#_______________________________________________________________________________________________________________________
class RNNsModelTuning(nn.Module):
    def __init__(self,num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat,num_numFamMembCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers, hidden_units,
                 rnn_type,
                 activation_func_act, activation_func_bi,
                 dropout_loc, dropout_withNOB, dropout_embedding, dropout_RNNs,
                 embed_size,
                 ):

        super(RNNsModelTuning, self).__init__()
        embed_size = embed_size
        # Batch normalization layers are included to help stabilize and speed up the training process.
        self.BatchNorm1d_size = 48  # Store BatchNorm1d_size as an instance variable
        self.rnn_type = rnn_type
        # Store num_continuous_features as an attribute
        self.num_continuous_features = num_continuous_features

        if activation_func_act == 'relu':
            self.activation_act = nn.ReLU()
        elif activation_func_act == 'tanh':
            self.activation_act = nn.Tanh()

        if activation_func_bi == 'relu':
            self.activation_binary = nn.ReLU()
        elif activation_func_bi == 'tanh':
            self.activation_binary = nn.Tanh()

        # Define embedding dimensions for all categorical features
        # Occupant Demographics
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 2)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 2)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 2)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        # Occupant Demographics
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the LSTM layers
        total_embedding_dim = sum([
            # Occupant Demographics
            self.embedding_dim_education,
            self.embedding_dim_employment,
            self.embedding_dim_gender,
            self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb,
            # Order columns
            self.embedding_dim_OCCinHH,
            # non-temporal TUS daily features
            self.embedding_dim_season,
            self.embedding_dim_weekend,
        ])
        input_size = total_embedding_dim + num_continuous_features
        #print("input_size:", input_size)

        # Dynamic LSTM layers initialization
        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.multiplier = 2 # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            if self.rnn_type == 'LSTM':
                rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            elif self.rnn_type == 'RNN':
                rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            else:
                raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', 'RNN'.")
            self.shared_rnns.append(rnn_layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * self.multiplier))
            self.dropouts.append(nn.Dropout(p=dropout_RNNs))
            input_size = hidden_size * self.multiplier

        # Activity output layer
        self.activity_batch_norm = nn.BatchNorm1d(input_size)
        self.activity_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_activity)  # Assuming bidirectional LSTM

        # Location output layer with Dropout: Dropout is used to help prevent overfitting
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.location_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_location)  # Assuming bidirectional LSTM

        # withNOBODY output layer with Dropout: Dropout is used to help prevent overfitting
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)
        self.withNOB_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_withNOB)  # Assuming bidirectional LSTM


        self.activity_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_activity)  # Assuming bidirectional LSTM

    def forward(self,  education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                OCCinHH_input,
                season_input, weekend_input,
                continuous_input):

        # Embeddings
        education_embedded = self.education_embedding(education_input).reshape(-1, 48, self.embedding_dim_education)
        #print("Initial Education Embedding: ", education_embedded)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 48,self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 48, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 48,self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 48,self.embedding_dim_numFamMemb)
        # Order columns
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 48, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((education_embedded,employment_embedded,gender_embedded, famTypology_embedded, numFamMemb_embedded,
                                           OCCinHH_embedded, season_embedded, weekend_embedded, continuous_input), dim=2)

        rnn_out = concatenated_features

        # Dynamic RNNs processing
        for rnn_layer, batch_norm, dropout in zip(self.shared_rnns, self.batch_norms, self.dropouts):
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = batch_norm(rnn_out.reshape(-1, rnn_out.shape[2])).reshape(rnn_out.size(0), -1, rnn_out.shape[2])
            # Dropout
            rnn_out = dropout(rnn_out) # Dropout applied here, controlled by dropout_RNNs parameter

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(self.activation_act(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        activity_output = activity_output.reshape(-1, 48, activity_output.shape[-1])  # Reshape to sequence form

        # Location with Dropout
        location_output = self.location_dropout(rnn_out)
        location_output = self.location_dense(self.activation_binary(location_output.contiguous().reshape(-1, location_output.shape[2])))
        location_output = location_output.reshape(-1, 48, location_output.shape[-1])  # Reshape to sequence form

        # WithNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(rnn_out)
        withNOB_output = self.withNOB_dense(self.activation_binary(withNOB_output.contiguous().reshape(-1, withNOB_output.shape[2])))
        withNOB_output = withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])  # Reshape to sequence form

        return activity_output, location_output, withNOB_output

# TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER
# TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER

import math
import torch
import torch.nn as nn
# TRANSFORMER: EXTRA ---------------------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# TRANSFORMER: MODEL FULL ---------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
# Import LayerNorm at the top
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TransformerModel(nn.Module):
    def __init__(self, num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat, num_numFamMembCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers,
                 dropout_loc = 0.5, dropout_withNOB = 0.5, dropout_embedding=0.5, dropout_transformer=0.5,
                 embed_size=50,
                 nhead=4):

        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.activation = nn.ReLU()
        self.nhead = nhead

        # Define embedding dimensions for all categorical features
        # Occupant Demographics
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 1)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 1)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 1)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the transformer layers
        total_embedding_dim = sum([
            self.embedding_dim_education,
            self.embedding_dim_employment,
            self.embedding_dim_gender,
            self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb,
            self.embedding_dim_OCCinHH,
            self.embedding_dim_season,
            self.embedding_dim_weekend,
        ])
        input_size = total_embedding_dim + num_continuous_features
        #print("hello, input size:", input_size)

        # Ensure input_size is divisible by nhead
        if input_size % nhead != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by nhead ({nhead}).")

        self.positional_encoding = PositionalEncoding(input_size)

        # Transformer encoder layer initialization
        self.encoder_layer = TransformerEncoderLayer(d_model=input_size, nhead=nhead, dropout=dropout_transformer, activation=self.activation, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_hidden_layers, )

        #self.batch_norm = nn.BatchNorm1d(input_size)
        self.layer_norm = LayerNorm(input_size)

        # Output layers for each task
        self.activity_dense = nn.Linear(input_size, output_dim_activity)
        self.location_dense = nn.Linear(input_size, output_dim_location)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

        # Dropout layers for output
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)

    def forward(self, education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                OCCinHH_input, season_input, weekend_input, continuous_input):

        # Embeddings
        education_embedded = self.education_embedding(education_input).reshape(-1, 48, self.embedding_dim_education)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 48, self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 48, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 48, self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 48, self.embedding_dim_numFamMemb)
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 48, self.embedding_dim_OCCinHH)
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((education_embedded, employment_embedded, gender_embedded, famTypology_embedded,numFamMemb_embedded,
                                           OCCinHH_embedded,
                                           season_embedded, weekend_embedded,
                                           continuous_input), dim=2)

        # Apply Positional Encoding
        concatenated_features = self.positional_encoding(concatenated_features)

        # Apply Transformer Encoder
        transformer_out = self.transformer_encoder(concatenated_features)

        # Batch Normalization
        #transformer_out = transformer_out.permute(0, 2, 1)  # Permute for batch normalization
        #transformer_out = self.batch_norm(transformer_out)
        #transformer_out = transformer_out.permute(0, 2, 1)  # Permute back to original shape
        # Layer Normalization
        transformer_out = self.layer_norm(transformer_out)

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(transformer_out)

        # Location with Dropout
        location_output = self.location_dropout(transformer_out)
        location_output = self.location_dense(location_output)

        # WithNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(transformer_out)
        withNOB_output = self.withNOB_dense(withNOB_output)

        return activity_output, location_output, withNOB_output

# TRANSFORMER: MODEL NO-EMBEDDING --------------------------------------------------------------------------------------
class TransformerModelNoEmbed(nn.Module):
    def __init__(self,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers,
                 dropout_loc=0.5, dropout_withNOB=0.5, dropout_transformer=0.5,
                 nhead=4,):

        super(TransformerModelNoEmbed, self).__init__()
        self.activation = nn.ReLU()
        input_size = num_continuous_features
        #print("input_size:", input_size)

        if input_size % nhead != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by nhead ({nhead}).")

        self.positional_encoding = PositionalEncoding(input_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=input_size, nhead=nhead, dropout=dropout_transformer, activation=self.activation, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_hidden_layers)
        self.batch_norm = nn.BatchNorm1d(input_size)

        self.activity_dense = nn.Linear(input_size, output_dim_activity)
        self.location_dense = nn.Linear(input_size, output_dim_location)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)

    def forward(self, continuous_input):
        continuous_input = self.positional_encoding(continuous_input)

        transformer_out = self.transformer_encoder(continuous_input)

        transformer_out = transformer_out.permute(0, 2, 1)  # Permute for batch normalization
        transformer_out = self.batch_norm(transformer_out)
        transformer_out = transformer_out.permute(0, 2, 1)  # Permute back to original shape

        activity_output = self.activity_dense(transformer_out)

        location_output = self.location_dropout(transformer_out)
        location_output = self.location_dense(location_output)

        withNOB_output = self.withNOB_dropout(transformer_out)
        withNOB_output = self.withNOB_dense(withNOB_output)

        return activity_output, location_output, withNOB_output

# TRANSFORMER TUNING----------------------------------------------------------------------------------------------------
from torch.nn import LayerNorm
class TransformerModelTuning(nn.Module):
    def __init__(self,num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat,num_numFamMembCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers,
                 dropout_loc, dropout_withNOB, dropout_embedding, dropout_transformer,
                 embed_size,
                 nhead,):

        super(TransformerModelTuning, self).__init__()
        self.embed_size = embed_size
        self.activation = nn.ReLU()
        self.nhead = nhead

        # Define embedding dimensions for all categorical features
        # Occupant Demographics
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 2)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 2)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 2)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the transformer layers
        total_embedding_dim = sum([
            self.embedding_dim_education,
            self.embedding_dim_employment,
            self.embedding_dim_gender,
            self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb,
            self.embedding_dim_OCCinHH,
            self.embedding_dim_season,
            self.embedding_dim_weekend,
        ])
        input_size = total_embedding_dim + num_continuous_features
        #print("input_size:", input_size)

        # Ensure input_size is divisible by nhead
        if input_size % self.nhead != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by nhead ({self.nhead}).")

        self.positional_encoding = PositionalEncoding(input_size)

        # Transformer encoder layer initialization
        self.encoder_layer = TransformerEncoderLayer(d_model=input_size,nhead=self.nhead,dropout=dropout_transformer,
                                                     activation=self.activation, batch_first=True) # multi-head attention mechanism & residual connections & Feed-Forward Network
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_hidden_layers,)

        # Batch Normalization
        #self.batch_norm = nn.BatchNorm1d(input_size)
        # Layer Normalization
        self.layer_norm = LayerNorm(input_size)

        # Output layers for each task
        self.activity_dense = nn.Linear(input_size, output_dim_activity)
        self.location_dense = nn.Linear(input_size, output_dim_location)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

        # Dropout layers for output
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)

    def forward(self, education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                OCCinHH_input, season_input, weekend_input, continuous_input):

        # Embeddings
        education_embedded = self.education_embedding(education_input).reshape(-1, 48, self.embedding_dim_education)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 48, self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 48, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 48, self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 48, self.embedding_dim_numFamMemb)
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 48, self.embedding_dim_OCCinHH)
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((education_embedded, employment_embedded, gender_embedded, famTypology_embedded,
                                           numFamMemb_embedded, OCCinHH_embedded, season_embedded,
                                           weekend_embedded, continuous_input), dim=2)

        concatenated_features = self.positional_encoding(concatenated_features)

        # Apply Transformer Encoder
        transformer_out = self.transformer_encoder(concatenated_features)

        # Batch Normalization
        #transformer_out = transformer_out.permute(0, 2, 1)  # Permute for batch normalization
        #transformer_out = self.batch_norm(transformer_out)
        #transformer_out = transformer_out.permute(0, 2, 1)  # Permute back to original shape

        # Layer Normalization
        transformer_out = self.layer_norm(transformer_out)

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(transformer_out)

        # Location with Dropout
        location_output = self.location_dropout(transformer_out)
        location_output = self.location_dense(location_output)

        # WithNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(transformer_out)
        withNOB_output = self.withNOB_dense(withNOB_output)

        return activity_output, location_output, withNOB_output

# N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS
# N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS

# N-HITS: NO-EMBEDDING & NO-LAYERS IN THE BLOCK ------------------------------------------------------------------------
class NHiTSBlockTuning(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, num_layers):
        super(NHiTSBlockTuning, self).__init__()
        self.backcast_mlp = self._build_mlp(input_size, output_size, hidden_units, num_layers)
        self.forecast_mlp = self._build_mlp(input_size, output_size, hidden_units, num_layers)

    def _build_mlp(self, input_size, output_size, hidden_units, num_layers):
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        backcast = self.backcast_mlp(x)
        forecast = self.forecast_mlp(x)
        return backcast, forecast
class NHiTSModelTuning(nn.Module):
    def __init__(self, num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat, num_numFamMembCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 dropout_embedding,
                 drop_loc, drop_NOB,
                 hidden_units, num_blocks, num_layers, embed_size):

        super(NHiTSModelTuning, self).__init__()
        self.embed_size = embed_size

        # Define Embedding Dimensions for all categorical features
        # Occupant Demographics
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 2)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 2)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 2)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the transformer layers
        total_embedding_dim = sum([
            self.embedding_dim_education, self.embedding_dim_employment, self.embedding_dim_gender,  self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb, self.embedding_dim_OCCinHH, self.embedding_dim_season, self.embedding_dim_weekend,
        ])
        input_size = total_embedding_dim + num_continuous_features

        # N-HiTS blocks
        self.blocks = nn.ModuleList(
            [NHiTSBlockTuning(input_size, input_size, hidden_units, num_layers) for _ in range(num_blocks)])

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(input_size)

        # Output layers with dropout
        self.dropout_output_location = nn.Dropout(p=drop_loc)
        self.dropout_output_withNOB = nn.Dropout(p=drop_NOB)

        # Output layers
        self.activity_dense = nn.Linear(input_size, output_dim_activity)
        self.location_dense = nn.Linear(input_size, output_dim_location)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

    def forward(self, education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                OCCinHH_input, season_input, weekend_input, continuous_input):
        # Embeddings
        education_embedded = self.education_embedding(education_input).reshape(-1, 48, self.embedding_dim_education)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 48, self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 48, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 48, self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 48, self.embedding_dim_numFamMemb)
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 48, self.embedding_dim_OCCinHH)
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((education_embedded, employment_embedded, gender_embedded, famTypology_embedded,numFamMemb_embedded,
                                           OCCinHH_embedded, season_embedded, weekend_embedded, continuous_input), dim=2)

        concatenated_features = self.dropout_embedding(concatenated_features)
        residual = concatenated_features
        forecast = torch.zeros_like(concatenated_features)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # Layer Normalization
        forecast = self.layer_norm(forecast)

        # Apply dropout before the final dense layers
        location_forecast = self.dropout_output_location(forecast)
        withNOB_forecast = self.dropout_output_withNOB(forecast)

        # Compute the final outputs
        activity_output = self.activity_dense(forecast)
        location_output = self.location_dense(location_forecast)
        withNOB_output = self.withNOB_dense(withNOB_forecast)

        return activity_output.reshape(-1, 48, activity_output.shape[-1]), \
            location_output.reshape(-1, 48, location_output.shape[-1]), \
            withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])

# N-HITS: BASE MODEL ---------------------------------------------------------------------------------------------------
class NHiTSBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, num_layers):
        super(NHiTSBlock, self).__init__()
        self.backcast_mlp = self._build_mlp(input_size, output_size, hidden_units, num_layers)
        self.forecast_mlp = self._build_mlp(input_size, output_size, hidden_units, num_layers)

    def _build_mlp(self, input_size, output_size, hidden_units, num_layers):
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        backcast = self.backcast_mlp(x)
        forecast = self.forecast_mlp(x)
        return backcast, forecast
class NHiTSModel(nn.Module):
    def __init__(self, num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat, num_numFamMembCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 dropout_embedding,
                 hidden_units, num_blocks, num_layers,
                 drop_act,drop_loc, drop_NOB,
                 embed_size=50):

        super(NHiTSModel, self).__init__()
        self.embed_size = embed_size

        # Define Embedding Dimensions for all categorical features
        # Occupant Demographics
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 2)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 2)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 2)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the transformer layers
        total_embedding_dim = sum([
            self.embedding_dim_education, self.embedding_dim_employment,
            self.embedding_dim_gender, self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb, self.embedding_dim_OCCinHH,
            self.embedding_dim_season, self.embedding_dim_weekend,
        ])
        input_size = total_embedding_dim + num_continuous_features

        # N-HiTS blocks
        self.blocks = nn.ModuleList(
            [NHiTSBlock(input_size, input_size, hidden_units, num_layers) for _ in range(num_blocks)])

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(input_size)

        # Output layers with dropout
        self.dropout_output_activity = nn.Dropout(p=drop_act)
        self.dropout_output_location = nn.Dropout(p=drop_loc)
        self.dropout_output_withNOB = nn.Dropout(p=drop_NOB)

        # Output layers
        self.activity_dense = nn.Linear(input_size, output_dim_activity)
        self.location_dense = nn.Linear(input_size, output_dim_location)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

    def forward(self, education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                OCCinHH_input, season_input, weekend_input, continuous_input):
        # Embeddings
        # Embeddings
        education_embedded = self.education_embedding(education_input).reshape(-1, 48, self.embedding_dim_education)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 48, self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 48, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 48, self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 48, self.embedding_dim_numFamMemb)
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 48, self.embedding_dim_OCCinHH)
        season_embedded = self.season_embedding(season_input).reshape(-1, 48, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 48, self.embedding_dim_weekend)

        # Concatenate all features
        concatenated_features = torch.cat((education_embedded, employment_embedded,
                                           gender_embedded, famTypology_embedded,
                                           numFamMemb_embedded,
                                           OCCinHH_embedded, season_embedded,
                                           weekend_embedded, continuous_input), dim=2)

        concatenated_features = self.dropout_embedding(concatenated_features)
        residual = concatenated_features
        forecast = torch.zeros_like(concatenated_features)  # Initialize forecast as a zero tensor with the same shape as concatenated_features

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # Layer Normalization
        forecast = self.layer_norm(forecast)

        # Apply dropout before the final dense layers
        activity_forecast = self.dropout_output_activity(forecast)
        location_forecast = self.dropout_output_location(forecast)
        withNOB_forecast = self.dropout_output_withNOB(forecast)

        # Compute the final outputs
        activity_output = self.activity_dense(activity_forecast)
        location_output = self.location_dense(location_forecast)
        withNOB_output = self.withNOB_dense(withNOB_forecast)

        return activity_output.reshape(-1, 48, activity_output.shape[-1]), \
            location_output.reshape(-1, 48, location_output.shape[-1]), \
            withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])

# N-HITS: NO-EMBEDDING IN THE BLOCK ------------------------------------------------------------------------------------

class NHiTSBlockNoEmbed(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, num_layers):
        super(NHiTSBlockNoEmbed, self).__init__()
        self.backcast_mlp = self._build_mlp(input_size, output_size, hidden_units, num_layers)
        self.forecast_mlp = self._build_mlp(input_size, output_size, hidden_units, num_layers)

    def _build_mlp(self, input_size, output_size, hidden_units, num_layers):
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        backcast = self.backcast_mlp(x)
        forecast = self.forecast_mlp(x)
        return backcast, forecast
class NHiTSModelNoEmbed(nn.Module):

    def __init__(self, input_size, output_dim_activity, output_dim_location, output_dim_withNOB, hidden_units, num_blocks, num_layers):
        super(NHiTSModelNoEmbed, self).__init__()
        self.blocks = nn.ModuleList([NHiTSBlockNoEmbed(input_size, input_size, hidden_units, num_layers) for _ in range(num_blocks)])

        # Activity output layer
        self.activity_dense = nn.Linear(input_size, output_dim_activity)

        # Location output layer
        self.location_dense = nn.Linear(input_size, output_dim_location)

        # WithNOBODY output layer
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

    def forward(self, x):
        residual = x
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # Compute the final outputs
        activity_output = self.activity_dense(forecast)
        location_output = self.location_dense(forecast)
        withNOB_output = self.withNOB_dense(forecast)

        return activity_output.reshape(-1, 48, activity_output.shape[-1]), \
            location_output.reshape(-1, 48, location_output.shape[-1]), \
            withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])

# N-HITS: NO-EMBEDDING & NO-LAYERS IN THE BLOCK ------------------------------------------------------------------------
class NHiTSBlockNoLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(NHiTSBlockNoLayers, self).__init__()
        self.backcast_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.forecast_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        backcast = self.backcast_mlp(x)
        forecast = self.forecast_mlp(x)
        return backcast, forecast
class NHiTSModelNoEmbedNoLayers(nn.Module):
    def __init__(self, input_size, output_dim_activity, output_dim_location, output_dim_withNOB, hidden_units,
                 num_blocks):
        super(NHiTSModelNoEmbedNoLayers, self).__init__()
        self.blocks = nn.ModuleList([NHiTSBlockNoLayers(input_size, input_size, hidden_units) for _ in range(num_blocks)])

        # Activity output layer
        self.activity_dense = nn.Linear(input_size, output_dim_activity)

        # Location output layer
        self.location_dense = nn.Linear(input_size, output_dim_location)

        # WithNOBODY output layer
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

    def forward(self, x):
        residual = x
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # Compute the final outputs
        activity_output = self.activity_dense(forecast)
        location_output = self.location_dense(forecast)
        withNOB_output = self.withNOB_dense(forecast)

        return activity_output.reshape(-1, 48, activity_output.shape[-1]), \
            location_output.reshape(-1, 48,location_output.shape[-1]), \
            withNOB_output.reshape(-1, 48, withNOB_output.shape[-1])
# MAMBA: NO-EMBEDDING ------------------------------------------------------------------------
