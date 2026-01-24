package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.exception.BlueWhaleException;
import com.seecoder.DataProcessing.po.User;
import com.seecoder.DataProcessing.repository.UserRepository;
import com.seecoder.DataProcessing.service.UserService;
import com.seecoder.DataProcessing.util.SecurityUtil;
import com.seecoder.DataProcessing.util.TokenUtil;
import com.seecoder.DataProcessing.vo.UserVO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Date;

/**
 * @Author: GaoZhaolong
 * @Date: 14:46 2023/11/26
 *
 * 注册登录功能实现
*/
@Service
public class UserServiceImpl implements UserService {

    @Autowired
    UserRepository userRepository;
    
    @Autowired
    TokenUtil tokenUtil;

    @Autowired
    SecurityUtil securityUtil;


    @Override
    public Boolean register(UserVO userVO) {
        User user = userRepository.findByPhone(userVO.getPhone());
        if (user != null) {
            throw BlueWhaleException.phoneAlreadyExists();
        }
        User newUser = userVO.toPO();
        newUser.setCreateTime(new Date());
        userRepository.save(newUser);
        return true;
    }

    @Override
    public String login(String phone, String password) {
        User user = userRepository.findByPhoneAndPassword(phone, password);
        if (user == null) {
            throw BlueWhaleException.phoneOrPasswordError();
        }
        return tokenUtil.getToken(user);
    }

    @Override
    public UserVO getInformation() {
        User user=securityUtil.getCurrentUser();
        return user.toVO();
    }

    @Override
    public Boolean updateInformation(UserVO userVO) {
        User user=securityUtil.getCurrentUser();
        if (userVO.getPassword()!=null){
            user.setPassword(userVO.getPassword());
        }
        if (userVO.getName()!=null){
            user.setName(userVO.getName());
        }

        userRepository.save(user);
        return true;
    }

}
