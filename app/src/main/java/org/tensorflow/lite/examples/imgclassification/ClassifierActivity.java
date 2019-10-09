/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imgclassification;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;


import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.SystemClock;
import android.util.Size;
import android.util.SparseArray;
import android.util.TypedValue;
import android.widget.Toast;
//import com.google.android.gms.vision

import androidx.annotation.RequiresApi;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import org.tensorflow.lite.examples.imgclassification.env.BorderedText;
import org.tensorflow.lite.examples.imgclassification.env.ImageUtils;
import org.tensorflow.lite.examples.imgclassification.env.Logger;
import org.tensorflow.lite.examples.imgclassification.tflite.Classifier;
import org.tensorflow.lite.examples.imgclassification.tflite.Classifier.Device;
import org.tensorflow.lite.examples.imgclassification.tflite.Classifier.Model;

@RequiresApi(api = Build.VERSION_CODES.KITKAT)
public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final boolean MAINTAIN_ASPECT = true;
  @SuppressLint("NewApi")
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap faceBitmap = null;
  private Bitmap copyFaceBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private BorderedText borderedText;

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap =
        Bitmap.createBitmap(
            //classifier.getImageSizeX(), classifier.getImageSizeY(), Config.ARGB_8888);
            480, 640, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth,
            previewHeight,
            480,
           640,
            sensorOrientation,
            MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    FaceDetector faceDetector = new
            FaceDetector.Builder(getApplicationContext()).setTrackingEnabled(false)
            .build();
    if (!faceDetector.isOperational())
    {
      //new AlertDialog.Builderv.getContext()).setMessage("Could not set up the face detector!").show();
      return;
    }
    new Thread(new Runnable()
      {
        @Override
        public void run() {
          SparseArray<Face> faces;
          Frame frame = new Frame.Builder().setBitmap(croppedBitmap).build();
          faces = faceDetector.detect(frame);
          if (faces.size()!=0)
          {
              Face thisFace =  faces.valueAt( faces.size() - 1);
              float x1 = thisFace.getPosition().x;
              float y1 = thisFace.getPosition().y;
              int faceW = (int) thisFace.getWidth();
              int faceH = (int) thisFace.getHeight();

              if (faceW > 224)
                faceW = 224;
              if (faceH > 224)
                 faceH = 224;

              float x2 = x1 + faceW;
              float y2 = y1 + faceH;

              faceBitmap= Bitmap.createBitmap(faceW, faceH, Config.RGB_565);
              Canvas canvas2 = new Canvas(faceBitmap);
              Rect src = new Rect((int) x1, (int) y1, (int) x2, (int) y2);
              Rect dst = new Rect(0, 0, faceW, faceH);
              canvas2.drawBitmap(croppedBitmap, src,dst, null);

              if (classifier != null) {
              final long startTime = SystemClock.uptimeMillis();
              final List<Classifier.Recognition> results = classifier.recognizeImage(faceBitmap);
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
              LOGGER.v("Detect: %s", results);
              cropCopyBitmap = Bitmap.createBitmap(faceBitmap);
              runOnUiThread(
                      new Runnable() {
                        @Override
                        public void run() {
                          showResultsInBottomSheet(results);
                          showFrameInfo(previewWidth + "x" + previewHeight);
                          showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                          showCameraResolution(canvas.getWidth() + "x" + canvas.getHeight());
                          showRotationInfo(String.valueOf(sensorOrientation));
                          showInference(lastProcessingTimeMs + "ms");
                        }
                      });
            }
          }
          else
            runOnUiThread(
                    new Runnable() {
                      @Override
                      public void run() {
                        showResultsInBottomSheet(null);
                        showFrameInfo(previewWidth + "x" + previewHeight);
                        showCropInfo("?" + "x" + "?");
                        showCameraResolution(canvas.getWidth() + "x" + canvas.getHeight());
                        showRotationInfo(String.valueOf(sensorOrientation));
                        showInference("NAN");
                      }
                    }
            );
          //faceDetector.release();
          readyForNextImage();
        }
      }).start();
  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (croppedBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU && model == Model.NATA_MODEL) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                .show();
          });
      return;
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }
  }
}
