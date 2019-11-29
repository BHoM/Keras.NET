using Numpy;
using System;
using System.Collections.Generic;
using System.Text;
using k = Keras;

namespace Keras.Layers
{
    /// <summary>
    /// Calculates the mean absolute error, also known as L1 loss.
    /// </summary>
    public class BaseLoss : Base
    {
    }

    /// <summary>
    /// Calculates the mean absolute error, also known as L1 loss.
    /// </summary>
    public class MeanAbsoluteError : BaseLoss
    {
        public MeanAbsoluteError(string reduction = null)
        {
            Parameters["reduction"] = reduction ?? "auto";
            Parameters["name"] = "mean_absolute_error";

            PyInstance = Keras.keras.losses.MeanAbsoluteError;
            Init();
        }
    }

    /// <summary>
    /// Calculates the mean squared error.
    /// </summary>
    public class MeanSquaredError : BaseLoss
    {
        public MeanSquaredError(string reduction = null)
        {
            Parameters["reduction"] = reduction ?? "auto";
            Parameters["name"] = "mean_squared_error";

            PyInstance = Keras.keras.losses.MeanSquaredError;
            Init();
        }
    }

    /// <summary>
    /// Computes the cross-entropy loss between true labels and predicted labels.
    /// </summary>
    public class BinaryCrossentropy : BaseLoss
    {
        public BinaryCrossentropy(string reduction = null)
        {
            Parameters["from_logits"] = true;
            Parameters["reduction"] = reduction ?? "auto";
            Parameters["name"] = "binary_crossentropy";

            PyInstance = Keras.keras.losses.BinaryCrossentropy;
            Init();
        }
    }

    /// <summary>
    /// Computes the sigmoid and the cross-entropy loss between true labels and predicted labels.
    /// </summary>
    public class BCEWithLogits : BaseLoss
    {
        public BCEWithLogits(string reduction = null)
        {
            Parameters["from_logits"] = false;
            Parameters["reduction"] = reduction ?? "auto";
            Parameters["name"] = "binary_crossentropy";

            PyInstance = Keras.keras.losses.BinaryCrossentropy;
            Init();
        }
    }

    /// <summary>
    /// Categoricals the crossentropy.
    /// </summary>
    public class CategoricalCrossentropy : BaseLoss
    {
        public CategoricalCrossentropy(string reduction = null)
        {
            Parameters["from_logits"] = true;
            Parameters["reduction"] = reduction ?? "auto";
            Parameters["name"] = "categorical_crossentropy";

            PyInstance = Keras.keras.losses.CategoricalCrossentropy;
            Init();
        }
    }

    /// <summary>
    /// Categoricals the crossentropy.
    /// </summary>
    public class NegLogLikelihood : BaseLoss
    {
        public NegLogLikelihood(string reduction = null)
        {
            Parameters["from_logits"] = false;
            Parameters["reduction"] = reduction ?? "auto";
            Parameters["name"] = "categorical_crossentropy";

            PyInstance = Keras.keras.losses.CategoricalCrossentropy;
            Init();
        }
    }
}
