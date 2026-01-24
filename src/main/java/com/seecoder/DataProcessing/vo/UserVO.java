package com.seecoder.DataProcessing.vo;

import com.seecoder.DataProcessing.enums.RoleEnum;
import com.seecoder.DataProcessing.po.User;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Date;

@Getter
@Setter
@NoArgsConstructor
public class UserVO {

    private Integer id;

    private String name;

    private String phone;

    private String password;

    private Date createTime;

    public User toPO(){
        User user=new User();
        user.setId(this.id);
        user.setName(this.name);
        user.setPhone(this.phone);
        user.setPassword(this.password);
        user.setCreateTime(this.createTime);
        return user;
    }
}
