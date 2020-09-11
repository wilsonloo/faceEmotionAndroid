package com.arcsoft.arcfacedemo.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Trace;

import com.arcsoft.arcfacedemo.R;
import com.tencent.bugly.crashreport.CrashReport;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public abstract class Classifier {
  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 5;

  /** The model type used for classification. */
  public enum Model {
    FLOAT_MOBILENET,
    QUANTIZED_MOBILENET,
    FLOAT_EFFICIENTNET,
    QUANTIZED_EFFICIENTNET
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** Image size along the x axis. */
  private final int imageSizeX;

  /** Image size along the y axis. */
  private final int imageSizeY;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** Input image TensorBuffer. */
  private TensorImage inputImageBuffer;

  /** Output probability TensorBuffer. */
  private final TensorBuffer outputProbabilityBuffer;

  /** Processer to apply post processing of the output probability. */
  private final TensorProcessor probabilityProcessor;

  private final Map<String, Integer> mEmotionResourceMap = new HashMap<>();

  // 识别结果
  public static class Recognition {
    private final String id;
    private final String title;
    private final Float confidence;

    public Recognition(final String id, final String title, final Float confidence) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
    }

    public String getTitle() { return this.title;}
    public Float getConfidence() {
      return this.confidence;
    }
  }

  // 内部构造函数，应该由工厂接口create创建实例
  protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
    mEmotionResourceMap.put("anger", R.mipmap.anger);
    mEmotionResourceMap.put("disgust", R.mipmap.disgust);
    mEmotionResourceMap.put("fear", R.mipmap.fear);
    mEmotionResourceMap.put("happy", R.mipmap.happy);
    mEmotionResourceMap.put("sad", R.mipmap.sad);

    // 加载tflite 模型
    String modelPath = getModelPath();
    tfliteModel = FileUtil.loadMappedFile(activity, modelPath);

    // 暂时只支持cpu配置
    if (device != Device.CPU) {
      throw new UnsupportedOperationException("device:" + device);
    }

    // 解析器的参数配置
    tfliteOptions.setNumThreads(numThreads);
    tflite = new Interpreter(tfliteModel, tfliteOptions);

    // 标签文件
    String labelPath = getLabelPath();
    labels = FileUtil.loadLabels(activity, labelPath);

    // 模型的输入数据结构信息
    int imageTensorIndex = 0;
    // 输入的图片尺寸信息
    int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
    imageSizeY = imageShape[1];
    imageSizeX = imageShape[2];

    // 输入的图片像素的元数据类型
    DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

    // 模型的输出数据结构信息（概率分布图）
    int probabilityTensorIndex = 0;
    int[] probabilityShape =
        tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
    DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

    // 构建输入图片的tensor
    inputImageBuffer = new TensorImage(imageDataType);

    // 模型输出的概率分布tensor
    outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

    // 输出的概率分布图，需要进行归一化

    probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
  }

  public List<Recognition> RecognizeImage(final Bitmap bitmap, int sensorOrientation) {
    Trace.beginSection("testRecognizeImage");

    // 输入数据
    Trace.beginSection("loadImage");
    inputImageBuffer = loadImage(bitmap, sensorOrientation);
    Trace.endSection();

    // 输出预测的概率分布
    Trace.beginSection("runInference");
    tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
    Trace.endSection();

    // 转换为 各个标签对应的概率
    Map<String /*标签*/, Float /*预测成该标签的概率*/> labeledProbability =
        new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
            .getMapWithFloatValue();

    Trace.endSection();

    // 获取前几项结果
    return getTopKProbability(labeledProbability, 1);
  }

  // 获取前K排名
  private static List<Recognition> getTopKProbability(Map<String /*标签*/, Float /*预测成该标签的概率*/> labelProb, int topK) {
    // 使用最大堆实现
    PriorityQueue<Recognition> pq =
        new PriorityQueue<>(
            MAX_RESULTS,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition o1, Recognition o2) {
                return o2.getConfidence().compareTo(o1.getConfidence());
              }
            });

      for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
        pq.add(new Recognition(""+entry.getKey(), entry.getKey(), entry.getValue()));
      }

      final ArrayList<Recognition> recognitionArrayList = new ArrayList<>();
      int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
      for (int k = 0; k < recognitionsSize; ++k){
        recognitionArrayList.add(pq.poll());
      }

      return recognitionArrayList;
  }

  // 加载图片并进行预处理(例如尺寸统一化、旋转等)
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    inputImageBuffer.load(bitmap);

    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRotation = sensorOrientation / 90;

    // TODO(b/143564309): Fuse ops inside ImageProcessor.
    ImageProcessor imageProcessor =
        new ImageProcessor.Builder()
            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
            .add(new Rot90Op(numRotation))
            .add(getPreprocessNormalizeOp())
            .build();
    return imageProcessor.process(inputImageBuffer);
  }

  // 获取tflite模型所在目录
  public abstract String getModelPath();

  // 获取分类标签所在目录
  public abstract String getLabelPath();

  /** Gets the TensorOperator to nomalize the input image in preprocessing. */
  protected abstract TensorOperator getPreprocessNormalizeOp();

  /**
   * Gets the TensorOperator to dequantize the output probability in post processing.
   *
   * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
   * essentially linear transformation). For float model, de-quantize is not required. But to
   * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
   * 1.0f, respectively.
   */
  protected abstract TensorOperator getPostprocessNormalizeOp();

  /**
   * 根据配置创建具体的 分类器 工厂方法设计模式： 只有一类抽象产品（abstract class Classifier），具体工厂（本匿名工厂）创建一个具体产品
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier Create(Activity activity, Model model, Device device, int numThreads)
      throws IOException {
      if (model == Model.FLOAT_MOBILENET) {
          return new ClassifierFloatMobileNet(activity, device, numThreads);
      } else {
          throw new UnsupportedOperationException("model:" + model);
      }
  }

    public Integer GetEmotionResourceId(String typeName)
        throws InvalidParameterException
    {
        Integer resource = mEmotionResourceMap.get(typeName);
        if(resource == null){
            throw new InvalidParameterException("unknown emotion type:"+typeName +":"+ typeName.length());
        }

        return resource;
    }
}
