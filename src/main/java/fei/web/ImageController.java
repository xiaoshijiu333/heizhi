package fei.web;

import fei.service.ImageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.commons.CommonsMultipartFile;

import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

/**
 * @author xiaoshijiu
 * @location HeiZhi/fei.web
 * @date 2019/3/6
 */
@Controller
public class ImageController {

    @Autowired
    private ImageService imageService;

    @RequestMapping("/html/image")
    @ResponseBody
    public List<String> image(@RequestParam("file") CommonsMultipartFile file, HttpSession session) throws IOException {

        if (file!=null){
            //1.新建接收存储目录
            String upload = session.getServletContext().getRealPath("upload");
            //2.路径转程序的文件
            File uploadfile = new File(upload);
            //3.判断文件是否存在，不存在则需要新建一个
            if (!uploadfile.exists()) {
                uploadfile.mkdirs();
            }
            //4.文件名处理（避免文件重名覆盖），使用UUID
            int index = file.getOriginalFilename().lastIndexOf(".");
            String exname = file.getOriginalFilename().substring(index);
            String uuid = UUID.randomUUID().toString();
            String fileName = uuid.replace("-", "") + exname;
            //5.最终具体文件路径变文件
            File newfile = new File(upload + "/" + fileName);
            //6.使用工具将文件转换过来
            file.transferTo(newfile);

            //请求服务层进行图片分析
            List<String> imageList = imageService.image(upload + "/" + fileName);

            return imageList;
        }
        return null;
    }
}
