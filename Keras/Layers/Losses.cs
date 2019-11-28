using Numpy;
using System;
using System.Collections.Generic;
using System.Text;
using k = Keras;

namespace Keras
{
    /// <summary>
    /// Calculates the mean absolute error, also known as L1 loss.
    /// </summary>
    public class MeanAbsoluteError : Base
    {
        public MeanAbsoluteError()
        {
            PyInstance = Keras.keras.losses.MeanAbsoluteError;
            Init();
        }
    }

    /// <summary>
    /// Calculates the mean squared error.
    /// </summary>
    public class MeanSquaredError : Base
    {
        public MeanSquaredError()
        {
            PyInstance = Keras.keras.losses.MeanSquaredError;
            Init();
        }
    }

    /// <summary>
    /// Computes the cross-entropy loss between true labels and predicted labels.
    /// </summary>
    public class BinaryCrossentropy : Base
    {
        public BinaryCrossentropy()
        {
            Parameters["from_logits"] = true;

            PyInstance = Keras.keras.losses.BinaryCrossentropy;
            Init();
        }
    }

    /// <summary>
    /// Computes the sigmoid and the cross-entropy loss between true labels and predicted labels.
    /// </summary>
    public class BCEWithLogits : Base
    {
        public BCEWithLogits()
        {
            Parameters["from_logits"] = false;

            PyInstance = Keras.keras.losses.BinaryCrossentropy;
            Init();
        }
    }

    /// <summary>
    /// Categoricals the crossentropy.
    /// </summary>
    public class CategoricalCrossentropy : Base
    {
        public CategoricalCrossentropy()
        {
            Parameters["from_logits"] = true;
        
            PyInstance = Keras.keras.losses.CategoricalCrossentropy;
            Init();
        }
    }

    /// <summary>
    /// Categoricals the crossentropy.
    /// </summary>
    public class NegLogLikelihood : Base
    {
        public NegLogLikelihood()
        {
            Parameters["from_logits"] = false;

            PyInstance = Keras.keras.losses.CategoricalCrossentropy;
            Init();
        }
    }
}
