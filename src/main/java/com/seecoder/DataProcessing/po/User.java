package com.seecoder.DataProcessing.po;

import com.seecoder.DataProcessing.enums.RoleEnum;
import com.seecoder.DataProcessing.vo.UserVO;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;
import java.util.Date;

@Getter
@Setter
@NoArgsConstructor
@Entity
public class User {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    @Column(name = "id")
    private Integer id;

    @Basic
    @Column(name = "name")
    private String name;

    @Basic
    @Column(name = "phone")
    private String phone;

    @Basic
    @Column(name = "password")
    private String password;

    //必须注意，在Java中用驼峰，在MySQL字段中用连字符_
    @Basic
    @Column(name = "create_time")
    private Date createTime;

    public UserVO toVO(){
        UserVO userVO=new UserVO();
        userVO.setId(this.id);
        userVO.setName(this.name);
        userVO.setPhone(this.phone);
        userVO.setPassword(this.password);
        userVO.setCreateTime(this.createTime);
        return userVO;
    }
}
