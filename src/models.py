import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel


class LenaRNN(nn.Module):
    def __init__(
        self,
        transactions_cat_features,
        embedding_projections,
        numerical_features,
        station_col_name="hydro_fixed_station_id_categorical",
        day_col_name="day_target_categorical",
        rnn_units=128,
        top_classifier_units=32,
    ):
        super().__init__()
        self.numerical_features = list(numerical_features)
        self._transaction_cat_embeddings = nn.ModuleList(
            [
                self._create_embedding_projection(*embedding_projections[feature])
                for feature in transactions_cat_features
            ]
        )

        self._product_embedding = self._create_embedding_projection(
            *embedding_projections[station_col_name], padding_idx=None
        )
        self._day_embedding = self._create_embedding_projection(*embedding_projections[day_col_name], padding_idx=None)

        self._gru = nn.GRU(
            input_size=sum([embedding_projections[x][1] for x in transactions_cat_features]) + len(numerical_features),
            hidden_size=rnn_units,
            batch_first=True,
            bidirectional=False,
        )

        self._hidden_size = rnn_units

        self._head = nn.Sequential(
            nn.Linear(
                rnn_units + embedding_projections[station_col_name][1] + embedding_projections[day_col_name][1],
                top_classifier_units,
            ),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(top_classifier_units, top_classifier_units),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(top_classifier_units, 1),
        )

    def forward(self, transactions_cat_features, product_feature, day_feature):
        batch_size = product_feature.shape[0]

        embeddings = [
            embedding(transactions_cat_features[:, :, i].long())
            for i, embedding in enumerate(self._transaction_cat_embeddings)
        ]
        embeddings.append(transactions_cat_features[:, :, -len(self.numerical_features) :].float())
        concated_embeddings = torch.cat(embeddings, dim=-1)

        _, last_hidden = self._gru(concated_embeddings)
        last_hidden = torch.reshape(last_hidden, shape=(batch_size, self._hidden_size))  # .permute(1, 2, 0)

        product_embed = self._product_embedding(product_feature.long())
        day_embed = self._day_embedding(day_feature.long())

        intermediate_concat = torch.cat([last_hidden, product_embed, day_embed], dim=-1)

        logit = self._head(intermediate_concat)

        return logit

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=False, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class LenaTrans(nn.Module):
    def __init__(
        self,
        cat_features,
        embeddings_projections,
        numerical_features,
        station_col_name="hydro_fixed_station_id_categorical",
        day_col_name="day_target_categorical",
        rnn_units=128,
        top_classifier_units=32,
    ):
        super().__init__()
        self.numerical_features = list(numerical_features)
        self._transaction_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embeddings_projections[feature]) for feature in cat_features]
        )

        self._product_embedding = self._create_embedding_projection(
            *embeddings_projections[station_col_name], padding_idx=None
        )
        self._day_embedding = self._create_embedding_projection(*embeddings_projections[day_col_name], padding_idx=None)

        self.config = AlbertConfig(
            3,  # not used
            embedding_size=sum([embeddings_projections[x][1] for x in cat_features]) + len(numerical_features),
            hidden_size=rnn_units,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=rnn_units,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=139,
            type_vocab_size=1,
            position_embedding_type="relative_key_query",
        )
        self.encoder = AlbertModel(self.config)

        self._hidden_size = rnn_units

        self._head = nn.Sequential(
            nn.Linear(
                rnn_units + embeddings_projections[station_col_name][1] + embeddings_projections[day_col_name][1],
                top_classifier_units,
            ),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(top_classifier_units, top_classifier_units),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(top_classifier_units, 1),
        )

    def forward(self, transactions_cat_features, product_feature, day_feature):
        batch_size = product_feature.shape[0]

        embeddings = [
            embedding(transactions_cat_features[:, :, i].long())
            for i, embedding in enumerate(self._transaction_cat_embeddings)
        ]
        embeddings.append(transactions_cat_features[:, :, -len(self.numerical_features) :].float())
        concated_embeddings = torch.cat(embeddings, dim=-1)

        seq_length = 139
        position_ids = torch.arange(seq_length, dtype=torch.long, device=concated_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))

        mask = torch.ones((batch_size, seq_length), device=concated_embeddings.device)

        for i in range(batch_size):
            for j in range(seq_length):
                if transactions_cat_features[i, j, :].sum() == 0:
                    mask[i, j] = 0

        encoded_layers = self.encoder(inputs_embeds=concated_embeddings, attention_mask=mask, position_ids=position_ids)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]

        product_embed = self._product_embedding(product_feature.long())
        day_embed = self._day_embedding(day_feature.long())

        intermediate_concat = torch.cat([sequence_output, product_embed, day_embed], dim=-1)

        logit = self._head(intermediate_concat)

        return logit

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=False, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class LenaTransExtra(nn.Module):
    def __init__(
        self,
        cat_features,
        embeddings_projections,
        numerical_features,
        target_cols,
        station_col_name="hydro_fixed_station_id_categorical",
        day_col_name="day_target_categorical",
        rnn_units=128,
        top_classifier_units=32,
    ):
        super().__init__()
        self.numerical_features = list(numerical_features)
        self.target_cols = target_cols
        self._transaction_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embeddings_projections[feature]) for feature in cat_features]
        )

        self._product_embedding = self._create_embedding_projection(
            *embeddings_projections[station_col_name], padding_idx=None
        )
        self._day_embedding = self._create_embedding_projection(*embeddings_projections[day_col_name], padding_idx=None)

        self.config = AlbertConfig(
            3,  # not used
            embedding_size=sum([embeddings_projections[x][1] for x in cat_features]) + len(numerical_features),
            hidden_size=rnn_units,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=rnn_units,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=139,
            type_vocab_size=1,
            position_embedding_type="relative_key_query",
        )
        self.encoder = AlbertModel(self.config)

        self._hidden_size = rnn_units

        self._head = nn.Sequential(
            nn.Linear(
                rnn_units + embeddings_projections[station_col_name][1] + embeddings_projections[day_col_name][1],
                top_classifier_units,
            ),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(top_classifier_units, top_classifier_units),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(top_classifier_units, len(self.target_cols)),
        )

    def forward(self, transactions_cat_features, product_feature, day_feature):
        batch_size = product_feature.shape[0]

        embeddings = [
            embedding(transactions_cat_features[:, :, i].long())
            for i, embedding in enumerate(self._transaction_cat_embeddings)
        ]
        embeddings.append(transactions_cat_features[:, :, -len(self.numerical_features) :].float())
        concated_embeddings = torch.cat(embeddings, dim=-1)

        seq_length = 139
        position_ids = torch.arange(seq_length, dtype=torch.long, device=concated_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))

        mask = torch.ones((batch_size, seq_length), device=concated_embeddings.device)

        for i in range(batch_size):
            for j in range(seq_length):
                if transactions_cat_features[i, j, :].sum() == 0:
                    mask[i, j] = 0

        encoded_layers = self.encoder(inputs_embeds=concated_embeddings, attention_mask=mask, position_ids=position_ids)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]

        product_embed = self._product_embedding(product_feature.long())
        day_embed = self._day_embedding(day_feature.long())

        intermediate_concat = torch.cat([sequence_output, product_embed, day_embed], dim=-1)

        logit = self._head(intermediate_concat)

        return logit

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=False, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
