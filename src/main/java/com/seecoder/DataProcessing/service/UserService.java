package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.vo.UserVO;

public interface UserService {
    Boolean register(UserVO userVO);

    String login(String phone,String password);

    UserVO getInformation();

    Boolean updateInformation(UserVO userVO);
}
