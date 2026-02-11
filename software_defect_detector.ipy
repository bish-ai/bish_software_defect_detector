{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP9fRslRb1S50H7jPQsJSeN",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bish-ai/bish_software_defect_detector/blob/main/software_defect_detector.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "data=pd.read_csv(\"https://storage.googleapis.com/kagglesdsdata/datasets/9419698/14739757/software_defect_prediction_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260211%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260211T183502Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=24080f042d5ef544ad0570e23603aafae193bf97d25e1be591cccbde173d19d68d98e52ad30219d264c78f9888c40318dfc77ec3ce0e1341c3cf50a4babc6a4ec745bc1e4d3633fbbb28ed983bfc1d8b7d0333e3815f31fc5cdfa438fe516ec0a5b0ab2e28c25a1bea300739d23e8fef56de0a41763e46d00a4ca3b78658425a6c618e52d81284cb46d3c8c3455668852b8fe56d3ced0996017cbad6ff896fa61fefd8a40c879bd2d9cba54cdbcceb088ec173fc508c54ac417231c1616a2e08dd8a8243a218ece0df6e8c780c2d0ed42903d069afcdf6a8b1717a0830930c8bb421c7d775eb8deaf77ebb07b6e269c3ed46383891eb411f41686339f53a1b91\")\n",
        "data.shape#pca\n",
        "data.isnull().sum()\n",
        "data.drop_duplicates()\n",
        "data.info()\n",
        "data.describe()\n",
        "data.corr()\n",
        "x=data.drop(columns=[\"defect\"])\n",
        "y=data[[\"defect\"]]\n",
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)\n",
        "x_train.head()\n",
        "from sklearn.preprocessing import RobustScaler\n",
        "rs_feauture=RobustScaler()\n",
        "rs_fit_transform_feauture=rs_feauture.fit_transform(x_train)\n",
        "rs_transform_feauture=rs_feauture.transform(x_test)\n",
        "from sklearn.decomposition import PCA\n",
        "pca_feauture=PCA(n_components=10)\n",
        "pca_fit_transform_feauture_rs=pca_feauture.fit_transform(rs_fit_transform_feauture)\n",
        "pca_transform_feauture_rs=pca_feauture.transform(rs_transform_feauture)\n",
        "from sklearn.preprocessing import PowerTransformer\n",
        "pt_feauture=PowerTransformer()\n",
        "pt_fit_pca=pt_feauture.fit_transform(pca_fit_transform_feauture_rs)\n",
        "pt_transform_pca=pt_feauture.transform(pca_transform_feauture_rs)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bi2EI_Jy-Caa",
        "outputId": "11449056-e889-4846-e656-f884494a31ea"
      },
      "execution_count": 129,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 60000 entries, 0 to 59999\n",
            "Data columns (total 23 columns):\n",
            " #   Column                      Non-Null Count  Dtype  \n",
            "---  ------                      --------------  -----  \n",
            " 0   lines_of_code               60000 non-null  int64  \n",
            " 1   cyclomatic_complexity       60000 non-null  int64  \n",
            " 2   num_functions               60000 non-null  int64  \n",
            " 3   num_classes                 60000 non-null  int64  \n",
            " 4   comment_density             60000 non-null  float64\n",
            " 5   code_churn                  60000 non-null  int64  \n",
            " 6   developer_experience_years  60000 non-null  int64  \n",
            " 7   num_developers              60000 non-null  int64  \n",
            " 8   commit_frequency            60000 non-null  int64  \n",
            " 9   bug_fix_commits             60000 non-null  int64  \n",
            " 10  past_defects                60000 non-null  int64  \n",
            " 11  test_coverage               60000 non-null  float64\n",
            " 12  duplication_percentage      60000 non-null  float64\n",
            " 13  avg_function_length         60000 non-null  int64  \n",
            " 14  depth_of_inheritance        60000 non-null  int64  \n",
            " 15  response_for_class          60000 non-null  int64  \n",
            " 16  coupling_between_objects    60000 non-null  int64  \n",
            " 17  lack_of_cohesion            60000 non-null  float64\n",
            " 18  build_failures              60000 non-null  int64  \n",
            " 19  static_analysis_warnings    60000 non-null  int64  \n",
            " 20  security_vulnerabilities    60000 non-null  int64  \n",
            " 21  performance_issues          60000 non-null  int64  \n",
            " 22  defect                      60000 non-null  int64  \n",
            "dtypes: float64(4), int64(19)\n",
            "memory usage: 10.5 MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_train.head()\n",
        "y_train.value_counts()\n",
        "y_train.sort_index()\n",
        "y_train.mean()\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "lr_model=LogisticRegression()\n",
        "lr_fit_model=lr_model.fit(pt_fit_pca,y_train)\n",
        "lr_pred_model=lr_model.predict(pt_transform_pca)\n",
        "from sklearn.metrics import accuracy_score\n",
        "accuracy_score(lr_pred_model,y_test)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lZaIcPziB_AH",
        "outputId": "4c9055bf-2aed-4fee-9229-7b72a5a459f4"
      },
      "execution_count": 137,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9688333333333333"
            ]
          },
          "metadata": {},
          "execution_count": 137
        }
      ]
    }
  ]
}
