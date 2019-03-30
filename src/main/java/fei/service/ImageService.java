package fei.service;

import org.springframework.stereotype.Service;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author xiaoshijiu
 * @location HeiZhi/fei.service
 * @date 2019/3/6
 */
@Service
public class ImageService {

    /**
     * @function: 响应web层请求，分析图片，返回预测概率
     * @params： 图片路径
     * @return： 预测概率集合
     */
    public List<String> image(String imageFile) throws IOException {

        //获取pb模型文件路径，必须要截取一下路径前面的/
        String path = ImageService.class.getClassLoader().getResource("tensor_model.pb").getPath();
        String subpath = path.substring(1);

        //获取pb模型文件的字节流
        byte[] graphDef = readAllBytesOrExit(Paths.get(subpath));

        //调用Tensorflow分析
        try (Tensor<Float> floatTensor = constructAndExecuteGraphToNormalizeImage(imageFile)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, floatTensor);
            //返回已经处理好的两位小数的百分数
            return changeFloat(labelProbabilities);
        }
    }

    /**
     * @function: 将图片从从磁盘加载到内存，重塑大小并获取各个像素点的RGB值，赋值给四维矩阵，并创建Tonsor
     * @params： 传入图片在磁盘中的地址
     * @return： 四维的Tensor
     */
    private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(String path) throws IOException {
        //加载图像，从磁盘加载到内存
        BufferedImage bimg = ImageIO.read(new File(path));
        //重塑图像大小
        BufferedImage tag = new BufferedImage(299, 299, BufferedImage.TYPE_INT_RGB);
        tag.getGraphics().drawImage(bimg, 0, 0, 299, 299, null);
        //定义矩阵 三维数组——>四维，每个像素点除以225
        float[][][][] data = new float[1][299][299][3];
        //获取RGB三个数，赋值给矩阵
        for (int i = 0; i < 299; i++) {
            for (int j = 0; j < 299; j++) {
                int rgb = tag.getRGB(i, j);
                data[0][i][j][0] = ((rgb & 0xff0000) >> 16) / 255.0f;
                data[0][i][j][1] = ((rgb & 0xff00) >> 8) / 255.0f;
                data[0][i][j][2] = (rgb & 0xff) / 255.0f;
            }
        }
        //创建Tensor
        Tensor<Float> tensor = (Tensor<Float>) Tensor.create(data);
        return tensor;
    }

    /**
     * @function: 调用Tensorflow进行分析
     * @params： 传入 四维的Tensor 和 pb模型文件的二进制流
     * @return： float[]数组，长度为7，分别对应各类预测概率
     */
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
                 Tensor<Float> result =
                         s.runner().feed("main_input:0", image).fetch("main_output/Softmax:0").run().get(0).expect(Float.class)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    /**
     * @function: 将float数组的每个数转换成百分数，并保留两位小数
     * @params： 传入 float数组
     * @return： 处理后的概率集合，包括了最大值的索引
     */
    private List<String> changeFloat(float[] labelProbabilities) {
        List<String> imageProbabilities = new ArrayList<>();
        for (Float labelProbability : labelProbabilities) {
            String s = String.format("%.2f%%", labelProbability * 100f);
            imageProbabilities.add(s);
        }
        //找到数组最大的值的索引
        String value = String.valueOf(maxIndex(labelProbabilities));
        imageProbabilities.add(value);
        return imageProbabilities;
    }

    /**
     * @function: 将pb模型文件加载成二进制流形式
     * @params： 传入 pb模型文件路径
     * @return： 模型文件的字节流
     */
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    /**
     * @function: 获取数组最大索引值
     * @params： 传入 数组
     * @return： int型 索引值
     */
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
}
