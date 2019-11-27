using Numpy;
using System;
using System.Collections.Generic;
using System.Text;
using k = Keras;

namespace Keras.Layers
{
    /// <summary>
    /// Calculates the mean squared error.
    /// </summary>
    public abstract class BaseLoss : Base
    {
        public abstract Func<NDarray, NDarray, NDarray> Call { get; }
    }

    /// <summary>
    /// Calculates the mean squared error.
    /// </summary>
    public class MeanSquaredError : BaseLoss
    {
        public override Func<NDarray, NDarray, NDarray> Call { get => k.Losses.MeanSquaredError; }
    }

    /// <summary>
    /// Binaries the crossentropy.
    /// </summary>
    public class BinaryCrossentropy : BaseLoss
    {
        public override Func<NDarray, NDarray, NDarray> Call { get => k.Losses.BinaryCrossentropy; }
    }

    /// <summary>
    /// Categoricals the crossentropy.
    /// </summary>
    public class CategoricalCrossentropy : BaseLoss
    {
        public override Func<NDarray, NDarray, NDarray> Call { get => k.Losses.CategoricalCrossentropy; }
    }

    ///// <summary>
    ///// Calculates the mean squared log error.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray MeanSquaredLogarithmicError(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "mean_squared_logarithmic_error", parameters));
    //}

    ///// <summary>
    ///// Calculates the Square Hinge
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray SquaredHinge(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "squared_hinge", parameters));
    //}

    ///// <summary>
    ///// Calculates the Hinge error.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray Hinge(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "hinge", parameters));
    //}

    ///// <summary>
    ///// Calculates the categorial hinge.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray CategoricalHinge(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "categorical_hinge", parameters));
    //}

    ///// <summary>
    ///// Logarithm of the hyperbolic cosine of the prediction error.
    ///// log(cosh(x)) is approximately equal to(x** 2) / 2 for small x and to abs(x) - log(2) for large x.This means that 'logcosh' works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray LogCosh(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "logcosh", parameters));
    //}

    ///// <summary>
    ///// Sparses the categorical crossentropy.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray SparseCategoricalCrossentropy(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_crossentropy", parameters));
    //}

    ///// <summary>
    ///// Binaries the crossentropy.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray BinaryCrossentropy(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "binary_crossentropy", parameters));
    //}

    ///// <summary>
    ///// Kullbacks the leibler divergence.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray KullbackLeiblerDivergence(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "kullback_leibler_divergence", parameters));
    //}

    ///// <summary>
    ///// Poissons the specified y true.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray Poisson(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "poisson", parameters));
    //}

    ///// <summary>
    ///// Cosines the proximity.
    ///// </summary>
    ///// <param name="y_true">tensor of true targets.</param>
    ///// <param name="y_pred">tensor of predicted targets.</param>
    ///// <returns></returns>
    //public static NDarray CosineProximity(NDarray y_true, NDarray y_pred)
    //{
    //    Dictionary<string, object> parameters = new Dictionary<string, object>();
    //    parameters["y_true"] = y_true;
    //    parameters["y_pred"] = y_pred;
    //    return new NDarray(InvokeStaticMethod(caller, "cosine_proximity", parameters));
    //}
}
