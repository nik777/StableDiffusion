import torch 

def get_unet_weights(org_weights):    
    unet_weights = {}

    # SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), # b 320 h/8 w/8
    unet_weights['encoder.0.0.weight'] = org_weights['model.diffusion_model.input_blocks.0.0.weight']
    unet_weights['encoder.0.0.bias']   = org_weights['model.diffusion_model.input_blocks.0.0.bias']

    # SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
    unet_weights['encoder.1.0.x_trans.0.weight']   = org_weights['model.diffusion_model.input_blocks.1.0.in_layers.0.weight']
    unet_weights['encoder.1.0.x_trans.0.bias']     = org_weights['model.diffusion_model.input_blocks.1.0.in_layers.0.bias']
    unet_weights['encoder.1.0.x_trans.2.weight']   = org_weights['model.diffusion_model.input_blocks.1.0.in_layers.2.weight']
    unet_weights['encoder.1.0.x_trans.2.bias']     = org_weights['model.diffusion_model.input_blocks.1.0.in_layers.2.bias']
    unet_weights['encoder.1.0.t_trans.1.weight']   = org_weights['model.diffusion_model.input_blocks.1.0.emb_layers.1.weight']
    unet_weights['encoder.1.0.t_trans.1.bias']     = org_weights['model.diffusion_model.input_blocks.1.0.emb_layers.1.bias']
    unet_weights['encoder.1.0.out_trans.0.weight'] = org_weights['model.diffusion_model.input_blocks.1.0.out_layers.0.weight']
    unet_weights['encoder.1.0.out_trans.0.bias']   = org_weights['model.diffusion_model.input_blocks.1.0.out_layers.0.bias']
    unet_weights['encoder.1.0.out_trans.2.weight'] = org_weights['model.diffusion_model.input_blocks.1.0.out_layers.3.weight']
    unet_weights['encoder.1.0.out_trans.2.bias']   = org_weights['model.diffusion_model.input_blocks.1.0.out_layers.3.bias']

    #----------------
    unet_weights['encoder.1.1.norm_conv1.0.weight']        = org_weights['model.diffusion_model.input_blocks.1.1.norm.weight']
    unet_weights['encoder.1.1.norm_conv1.0.bias']          = org_weights['model.diffusion_model.input_blocks.1.1.norm.bias']
    unet_weights['encoder.1.1.norm_conv1.1.weight']        = org_weights['model.diffusion_model.input_blocks.1.1.proj_in.weight']
    unet_weights['encoder.1.1.norm_conv1.1.bias']          = org_weights['model.diffusion_model.input_blocks.1.1.proj_in.bias']
    unet_weights['encoder.1.1.norm_self_att.0.weight']     = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight']
    unet_weights['encoder.1.1.norm_self_att.0.bias']       = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias']
    unet_weights['encoder.1.1.norm_self_att.1.w_o.weight'] = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight'] 
    unet_weights['encoder.1.1.norm_self_att.1.w_o.bias']   = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['encoder.1.1.norm.weight']                = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight']
    unet_weights['encoder.1.1.norm.bias']                  = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias']
    unet_weights['encoder.1.1.cross_att.w_q.weight']       = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['encoder.1.1.cross_att.w_k.weight']       = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['encoder.1.1.cross_att.w_v.weight']       = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['encoder.1.1.cross_att.w_o.weight']       = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['encoder.1.1.cross_att.w_o.bias']         = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['encoder.1.1.geglu.0.weight']             = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight']
    unet_weights['encoder.1.1.geglu.0.bias']               = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias']
    unet_weights['encoder.1.1.geglu.1.fc1.weight']         = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['encoder.1.1.geglu.1.fc1.bias']           = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['encoder.1.1.geglu.1.fc2.weight']         = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['encoder.1.1.geglu.1.fc2.bias']           = org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['encoder.1.1.conv_out.weight']            = org_weights['model.diffusion_model.input_blocks.1.1.proj_out.weight']
    unet_weights['encoder.1.1.conv_out.bias']              = org_weights['model.diffusion_model.input_blocks.1.1.proj_out.bias']
    unet_weights['encoder.1.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)
                
    #-----------------------    
    # SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),         
    unet_weights['encoder.2.0.x_trans.0.weight']   = org_weights['model.diffusion_model.input_blocks.2.0.in_layers.0.weight']
    unet_weights['encoder.2.0.x_trans.0.bias']     = org_weights['model.diffusion_model.input_blocks.2.0.in_layers.0.bias']
    unet_weights['encoder.2.0.x_trans.2.weight']   = org_weights['model.diffusion_model.input_blocks.2.0.in_layers.2.weight']
    unet_weights['encoder.2.0.x_trans.2.bias']     = org_weights['model.diffusion_model.input_blocks.2.0.in_layers.2.bias']
    unet_weights['encoder.2.0.t_trans.1.weight']   = org_weights['model.diffusion_model.input_blocks.2.0.emb_layers.1.weight']
    unet_weights['encoder.2.0.t_trans.1.bias']     = org_weights['model.diffusion_model.input_blocks.2.0.emb_layers.1.bias']
    unet_weights['encoder.2.0.out_trans.0.weight'] = org_weights['model.diffusion_model.input_blocks.2.0.out_layers.0.weight']
    unet_weights['encoder.2.0.out_trans.0.bias']   = org_weights['model.diffusion_model.input_blocks.2.0.out_layers.0.bias']
    unet_weights['encoder.2.0.out_trans.2.weight'] = org_weights['model.diffusion_model.input_blocks.2.0.out_layers.3.weight']
    unet_weights['encoder.2.0.out_trans.2.bias']   = org_weights['model.diffusion_model.input_blocks.2.0.out_layers.3.bias']
                
    unet_weights['encoder.2.1.norm_conv1.0.weight']        = org_weights['model.diffusion_model.input_blocks.2.1.norm.weight']
    unet_weights['encoder.2.1.norm_conv1.0.bias']          = org_weights['model.diffusion_model.input_blocks.2.1.norm.bias']
    unet_weights['encoder.2.1.norm_conv1.1.weight']        = org_weights['model.diffusion_model.input_blocks.2.1.proj_in.weight']
    unet_weights['encoder.2.1.norm_conv1.1.bias']          = org_weights['model.diffusion_model.input_blocks.2.1.proj_in.bias']
    unet_weights['encoder.2.1.norm_self_att.0.weight']     = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.weight']
    unet_weights['encoder.2.1.norm_self_att.0.bias']       = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.bias']
    unet_weights['encoder.2.1.norm_self_att.1.w_o.weight'] = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['encoder.2.1.norm_self_att.1.w_o.bias']   = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['encoder.2.1.norm.weight']                = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.weight']
    unet_weights['encoder.2.1.norm.bias']                  = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.bias']
    unet_weights['encoder.2.1.cross_att.w_q.weight']       = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['encoder.2.1.cross_att.w_k.weight']       = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['encoder.2.1.cross_att.w_v.weight']       = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['encoder.2.1.cross_att.w_o.weight']       = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['encoder.2.1.cross_att.w_o.bias']         = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['encoder.2.1.geglu.0.weight']             = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.weight']
    unet_weights['encoder.2.1.geglu.0.bias']               = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.bias']
    unet_weights['encoder.2.1.geglu.1.fc1.weight']         = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['encoder.2.1.geglu.1.fc1.bias']           = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['encoder.2.1.geglu.1.fc2.weight']         = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['encoder.2.1.geglu.1.fc2.bias']           = org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['encoder.2.1.conv_out.weight']            = org_weights['model.diffusion_model.input_blocks.2.1.proj_out.weight']
    unet_weights['encoder.2.1.conv_out.bias']              = org_weights['model.diffusion_model.input_blocks.2.1.proj_out.bias']
    unet_weights['encoder.2.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)), # b 320 h/16 w/16
    unet_weights['encoder.3.0.weight']                     = org_weights['model.diffusion_model.input_blocks.3.0.op.weight']
    unet_weights['encoder.3.0.bias']                       = org_weights['model.diffusion_model.input_blocks.3.0.op.bias']


    # SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),          # b 640 h/16 w/16              
    unet_weights['encoder.4.0.x_trans.0.weight']         = org_weights['model.diffusion_model.input_blocks.4.0.in_layers.0.weight']
    unet_weights['encoder.4.0.x_trans.0.bias']           = org_weights['model.diffusion_model.input_blocks.4.0.in_layers.0.bias']
    unet_weights['encoder.4.0.x_trans.2.weight']         = org_weights['model.diffusion_model.input_blocks.4.0.in_layers.2.weight']
    unet_weights['encoder.4.0.x_trans.2.bias']           = org_weights['model.diffusion_model.input_blocks.4.0.in_layers.2.bias']
    unet_weights['encoder.4.0.t_trans.1.weight']         = org_weights['model.diffusion_model.input_blocks.4.0.emb_layers.1.weight']
    unet_weights['encoder.4.0.t_trans.1.bias']           = org_weights['model.diffusion_model.input_blocks.4.0.emb_layers.1.bias']
    unet_weights['encoder.4.0.out_trans.0.weight']       = org_weights['model.diffusion_model.input_blocks.4.0.out_layers.0.weight']
    unet_weights['encoder.4.0.out_trans.0.bias']         = org_weights['model.diffusion_model.input_blocks.4.0.out_layers.0.bias']
    unet_weights['encoder.4.0.out_trans.2.weight']       = org_weights['model.diffusion_model.input_blocks.4.0.out_layers.3.weight']
    unet_weights['encoder.4.0.out_trans.2.bias']         = org_weights['model.diffusion_model.input_blocks.4.0.out_layers.3.bias']
    unet_weights['encoder.4.0.resize.weight']            = org_weights['model.diffusion_model.input_blocks.4.0.skip_connection.weight']
    unet_weights['encoder.4.0.resize.bias']              = org_weights['model.diffusion_model.input_blocks.4.0.skip_connection.bias']

    unet_weights['encoder.4.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.input_blocks.4.1.norm.weight']
    unet_weights['encoder.4.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.input_blocks.4.1.norm.bias']
    unet_weights['encoder.4.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.input_blocks.4.1.proj_in.weight']
    unet_weights['encoder.4.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.input_blocks.4.1.proj_in.bias']
    unet_weights['encoder.4.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight']
    unet_weights['encoder.4.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias']
    unet_weights['encoder.4.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['encoder.4.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['encoder.4.1.norm.weight']                             = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight']
    unet_weights['encoder.4.1.norm.bias']                               = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias']
    unet_weights['encoder.4.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['encoder.4.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['encoder.4.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['encoder.4.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['encoder.4.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['encoder.4.1.geglu.0.weight']                          = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight']
    unet_weights['encoder.4.1.geglu.0.bias']                            = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias']
    unet_weights['encoder.4.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['encoder.4.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['encoder.4.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['encoder.4.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['encoder.4.1.conv_out.weight']                         = org_weights['model.diffusion_model.input_blocks.4.1.proj_out.weight']
    unet_weights['encoder.4.1.conv_out.bias']                           = org_weights['model.diffusion_model.input_blocks.4.1.proj_out.bias']
    unet_weights['encoder.4.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
    unet_weights['encoder.5.0.x_trans.0.weight']   = org_weights['model.diffusion_model.input_blocks.5.0.in_layers.0.weight']
    unet_weights['encoder.5.0.x_trans.0.bias']     = org_weights['model.diffusion_model.input_blocks.5.0.in_layers.0.bias']
    unet_weights['encoder.5.0.x_trans.2.weight']   = org_weights['model.diffusion_model.input_blocks.5.0.in_layers.2.weight']
    unet_weights['encoder.5.0.x_trans.2.bias']     = org_weights['model.diffusion_model.input_blocks.5.0.in_layers.2.bias']
    unet_weights['encoder.5.0.t_trans.1.weight']   = org_weights['model.diffusion_model.input_blocks.5.0.emb_layers.1.weight']
    unet_weights['encoder.5.0.t_trans.1.bias']     = org_weights['model.diffusion_model.input_blocks.5.0.emb_layers.1.bias']
    unet_weights['encoder.5.0.out_trans.0.weight'] = org_weights['model.diffusion_model.input_blocks.5.0.out_layers.0.weight']
    unet_weights['encoder.5.0.out_trans.0.bias']   = org_weights['model.diffusion_model.input_blocks.5.0.out_layers.0.bias']
    unet_weights['encoder.5.0.out_trans.2.weight'] = org_weights['model.diffusion_model.input_blocks.5.0.out_layers.3.weight']
    unet_weights['encoder.5.0.out_trans.2.bias']   = org_weights['model.diffusion_model.input_blocks.5.0.out_layers.3.bias']

    unet_weights['encoder.5.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.input_blocks.5.1.norm.weight']
    unet_weights['encoder.5.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.input_blocks.5.1.norm.bias']
    unet_weights['encoder.5.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.input_blocks.5.1.proj_in.weight']
    unet_weights['encoder.5.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.input_blocks.5.1.proj_in.bias']
    unet_weights['encoder.5.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight']
    unet_weights['encoder.5.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias']
    unet_weights['encoder.5.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['encoder.5.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['encoder.5.1.norm.weight']                             = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight']
    unet_weights['encoder.5.1.norm.bias']                               = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias']
    unet_weights['encoder.5.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['encoder.5.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['encoder.5.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['encoder.5.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['encoder.5.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['encoder.5.1.geglu.0.weight']                          = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight']
    unet_weights['encoder.5.1.geglu.0.bias']                            = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias']
    unet_weights['encoder.5.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['encoder.5.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['encoder.5.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['encoder.5.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['encoder.5.1.conv_out.weight']                         = org_weights['model.diffusion_model.input_blocks.5.1.proj_out.weight']
    unet_weights['encoder.5.1.conv_out.bias']                           = org_weights['model.diffusion_model.input_blocks.5.1.proj_out.bias']
    unet_weights['encoder.5.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)), # b 640 h/32 w/32
    unet_weights['encoder.6.0.weight']                    = org_weights['model.diffusion_model.input_blocks.6.0.op.weight']
    unet_weights['encoder.6.0.bias']                      = org_weights['model.diffusion_model.input_blocks.6.0.op.bias']
                
    # SwitchSequential(ResidualBlock(640, 1280),  AttentionBlock(8, 160)),        # b 1280 h/32 w/32
    unet_weights['encoder.7.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.input_blocks.7.0.in_layers.0.weight']
    unet_weights['encoder.7.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.input_blocks.7.0.in_layers.0.bias']
    unet_weights['encoder.7.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.input_blocks.7.0.in_layers.2.weight']
    unet_weights['encoder.7.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.input_blocks.7.0.in_layers.2.bias']
    unet_weights['encoder.7.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.input_blocks.7.0.emb_layers.1.weight']
    unet_weights['encoder.7.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.input_blocks.7.0.emb_layers.1.bias']
    unet_weights['encoder.7.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.input_blocks.7.0.out_layers.0.weight']
    unet_weights['encoder.7.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.input_blocks.7.0.out_layers.0.bias']
    unet_weights['encoder.7.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.input_blocks.7.0.out_layers.3.weight']
    unet_weights['encoder.7.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.input_blocks.7.0.out_layers.3.bias']
    unet_weights['encoder.7.0.resize.weight']                           = org_weights['model.diffusion_model.input_blocks.7.0.skip_connection.weight']
    unet_weights['encoder.7.0.resize.bias']                             = org_weights['model.diffusion_model.input_blocks.7.0.skip_connection.bias']             

    unet_weights['encoder.7.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.input_blocks.7.1.norm.weight']
    unet_weights['encoder.7.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.input_blocks.7.1.norm.bias']
    unet_weights['encoder.7.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.input_blocks.7.1.proj_in.weight']
    unet_weights['encoder.7.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.input_blocks.7.1.proj_in.bias']
    unet_weights['encoder.7.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight']
    unet_weights['encoder.7.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias']
    unet_weights['encoder.7.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['encoder.7.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['encoder.7.1.norm.weight']                             = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight']
    unet_weights['encoder.7.1.norm.bias']                               = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias']
    unet_weights['encoder.7.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['encoder.7.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['encoder.7.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['encoder.7.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['encoder.7.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['encoder.7.1.geglu.0.weight']                          = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight']
    unet_weights['encoder.7.1.geglu.0.bias']                            = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias']
    unet_weights['encoder.7.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['encoder.7.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['encoder.7.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['encoder.7.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['encoder.7.1.conv_out.weight']                         = org_weights['model.diffusion_model.input_blocks.7.1.proj_out.weight']
    unet_weights['encoder.7.1.conv_out.bias']                           = org_weights['model.diffusion_model.input_blocks.7.1.proj_out.bias']
    unet_weights['encoder.7.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),

    unet_weights['encoder.8.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.input_blocks.8.0.in_layers.0.weight']
    unet_weights['encoder.8.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.input_blocks.8.0.in_layers.0.bias']
    unet_weights['encoder.8.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.input_blocks.8.0.in_layers.2.weight']
    unet_weights['encoder.8.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.input_blocks.8.0.in_layers.2.bias']
    unet_weights['encoder.8.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.input_blocks.8.0.emb_layers.1.weight']
    unet_weights['encoder.8.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.input_blocks.8.0.emb_layers.1.bias']
    unet_weights['encoder.8.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.input_blocks.8.0.out_layers.0.weight']
    unet_weights['encoder.8.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.input_blocks.8.0.out_layers.0.bias']
    unet_weights['encoder.8.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.input_blocks.8.0.out_layers.3.weight']
    unet_weights['encoder.8.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.input_blocks.8.0.out_layers.3.bias']
    unet_weights['encoder.8.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.input_blocks.8.1.norm.weight']
    unet_weights['encoder.8.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.input_blocks.8.1.norm.bias']
    unet_weights['encoder.8.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.input_blocks.8.1.proj_in.weight']
    unet_weights['encoder.8.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.input_blocks.8.1.proj_in.bias']
    unet_weights['encoder.8.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight']
    unet_weights['encoder.8.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias']
    unet_weights['encoder.8.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['encoder.8.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['encoder.8.1.norm.weight']                             = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight']
    unet_weights['encoder.8.1.norm.bias']                               = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias']
    unet_weights['encoder.8.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['encoder.8.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['encoder.8.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['encoder.8.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['encoder.8.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['encoder.8.1.geglu.0.weight']                          = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight']
    unet_weights['encoder.8.1.geglu.0.bias']                            = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias']
    unet_weights['encoder.8.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['encoder.8.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['encoder.8.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['encoder.8.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['encoder.8.1.conv_out.weight']                         = org_weights['model.diffusion_model.input_blocks.8.1.proj_out.weight']
    unet_weights['encoder.8.1.conv_out.bias']                           = org_weights['model.diffusion_model.input_blocks.8.1.proj_out.bias']
    unet_weights['encoder.8.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)
                
    # SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)), # b 1280 h/64 w/64
    unet_weights['encoder.9.0.weight']                  = org_weights['model.diffusion_model.input_blocks.9.0.op.weight']
    unet_weights['encoder.9.0.bias']                    = org_weights['model.diffusion_model.input_blocks.9.0.op.bias']

    # SwitchSequential(ResidualBlock(1280, 1280)),
    unet_weights['encoder.10.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.input_blocks.10.0.in_layers.0.weight']
    unet_weights['encoder.10.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.input_blocks.10.0.in_layers.0.bias']
    unet_weights['encoder.10.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.input_blocks.10.0.in_layers.2.weight']
    unet_weights['encoder.10.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.input_blocks.10.0.in_layers.2.bias']
    unet_weights['encoder.10.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.input_blocks.10.0.emb_layers.1.weight']
    unet_weights['encoder.10.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.input_blocks.10.0.emb_layers.1.bias']
    unet_weights['encoder.10.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.input_blocks.10.0.out_layers.0.weight']
    unet_weights['encoder.10.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.input_blocks.10.0.out_layers.0.bias']
    unet_weights['encoder.10.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.input_blocks.10.0.out_layers.3.weight']
    unet_weights['encoder.10.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.input_blocks.10.0.out_layers.3.bias']
    # SwitchSequential(ResidualBlock(1280, 1280))
                
    unet_weights['encoder.11.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.input_blocks.11.0.in_layers.0.weight']
    unet_weights['encoder.11.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.input_blocks.11.0.in_layers.0.bias']
    unet_weights['encoder.11.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.input_blocks.11.0.in_layers.2.weight']
    unet_weights['encoder.11.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.input_blocks.11.0.in_layers.2.bias']
    unet_weights['encoder.11.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.input_blocks.11.0.emb_layers.1.weight']
    unet_weights['encoder.11.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.input_blocks.11.0.emb_layers.1.bias']
    unet_weights['encoder.11.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.input_blocks.11.0.out_layers.0.weight']
    unet_weights['encoder.11.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.input_blocks.11.0.out_layers.0.bias']
    unet_weights['encoder.11.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.input_blocks.11.0.out_layers.3.weight']
    unet_weights['encoder.11.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.input_blocks.11.0.out_layers.3.bias']


    # ResidualBlock(1280, 1280), 
    unet_weights['bottleneck.0.x_trans.0.weight']   = org_weights['model.diffusion_model.middle_block.0.in_layers.0.weight']
    unet_weights['bottleneck.0.x_trans.0.bias']     = org_weights['model.diffusion_model.middle_block.0.in_layers.0.bias']
    unet_weights['bottleneck.0.x_trans.2.weight']   = org_weights['model.diffusion_model.middle_block.0.in_layers.2.weight']
    unet_weights['bottleneck.0.x_trans.2.bias']     = org_weights['model.diffusion_model.middle_block.0.in_layers.2.bias']
    unet_weights['bottleneck.0.t_trans.1.weight']   = org_weights['model.diffusion_model.middle_block.0.emb_layers.1.weight']
    unet_weights['bottleneck.0.t_trans.1.bias']     = org_weights['model.diffusion_model.middle_block.0.emb_layers.1.bias']
    unet_weights['bottleneck.0.out_trans.0.weight'] = org_weights['model.diffusion_model.middle_block.0.out_layers.0.weight']
    unet_weights['bottleneck.0.out_trans.0.bias']   = org_weights['model.diffusion_model.middle_block.0.out_layers.0.bias']
    unet_weights['bottleneck.0.out_trans.2.weight'] = org_weights['model.diffusion_model.middle_block.0.out_layers.3.weight']
    unet_weights['bottleneck.0.out_trans.2.bias']   = org_weights['model.diffusion_model.middle_block.0.out_layers.3.bias']
                
    # AttentionBlock(8, 160),              
    unet_weights['bottleneck.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.middle_block.1.norm.weight']
    unet_weights['bottleneck.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.middle_block.1.norm.bias']
    unet_weights['bottleneck.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.middle_block.1.proj_in.weight']
    unet_weights['bottleneck.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.middle_block.1.proj_in.bias']
    unet_weights['bottleneck.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight']
    unet_weights['bottleneck.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias']
    unet_weights['bottleneck.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['bottleneck.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['bottleneck.1.norm.weight']                             = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight']
    unet_weights['bottleneck.1.norm.bias']                               = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias']
    unet_weights['bottleneck.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['bottleneck.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['bottleneck.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['bottleneck.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['bottleneck.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['bottleneck.1.geglu.0.weight']                          = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight']
    unet_weights['bottleneck.1.geglu.0.bias']                            = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias']
    unet_weights['bottleneck.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['bottleneck.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['bottleneck.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['bottleneck.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['bottleneck.1.conv_out.weight']                         = org_weights['model.diffusion_model.middle_block.1.proj_out.weight']
    unet_weights['bottleneck.1.conv_out.bias']                           = org_weights['model.diffusion_model.middle_block.1.proj_out.bias']
    unet_weights['bottleneck.1.norm_self_att.1.w_qkv.weight']            = torch.cat((org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight'],
                                                                                    org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight'],
                                                                                    org_weights['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)
                
    # ResidualBlock(1280, 1280)             
    unet_weights['bottleneck.2.x_trans.0.weight']   = org_weights['model.diffusion_model.middle_block.2.in_layers.0.weight']
    unet_weights['bottleneck.2.x_trans.0.bias']     = org_weights['model.diffusion_model.middle_block.2.in_layers.0.bias']
    unet_weights['bottleneck.2.x_trans.2.weight']   = org_weights['model.diffusion_model.middle_block.2.in_layers.2.weight']
    unet_weights['bottleneck.2.x_trans.2.bias']     = org_weights['model.diffusion_model.middle_block.2.in_layers.2.bias']
    unet_weights['bottleneck.2.t_trans.1.weight']   = org_weights['model.diffusion_model.middle_block.2.emb_layers.1.weight']
    unet_weights['bottleneck.2.t_trans.1.bias']     = org_weights['model.diffusion_model.middle_block.2.emb_layers.1.bias']
    unet_weights['bottleneck.2.out_trans.0.weight'] = org_weights['model.diffusion_model.middle_block.2.out_layers.0.weight']
    unet_weights['bottleneck.2.out_trans.0.bias']   = org_weights['model.diffusion_model.middle_block.2.out_layers.0.bias']
    unet_weights['bottleneck.2.out_trans.2.weight'] = org_weights['model.diffusion_model.middle_block.2.out_layers.3.weight']
    unet_weights['bottleneck.2.out_trans.2.bias']   = org_weights['model.diffusion_model.middle_block.2.out_layers.3.bias']

    #SwitchSequential(ResidualBlock(2560, 1280)),
    unet_weights['decoder.0.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.0.0.in_layers.0.weight']
    unet_weights['decoder.0.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.0.0.in_layers.0.bias']
    unet_weights['decoder.0.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.0.0.in_layers.2.weight']
    unet_weights['decoder.0.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.0.0.in_layers.2.bias']
    unet_weights['decoder.0.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.0.0.emb_layers.1.weight']
    unet_weights['decoder.0.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.0.0.emb_layers.1.bias']
    unet_weights['decoder.0.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.0.0.out_layers.0.weight']
    unet_weights['decoder.0.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.0.0.out_layers.0.bias']
    unet_weights['decoder.0.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.0.0.out_layers.3.weight']
    unet_weights['decoder.0.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.0.0.out_layers.3.bias']
    unet_weights['decoder.0.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.0.0.skip_connection.weight']
    unet_weights['decoder.0.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.0.0.skip_connection.bias']     

    # SwitchSequential(ResidualBlock(2560, 1280)),
    unet_weights['decoder.1.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.1.0.in_layers.0.weight']
    unet_weights['decoder.1.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.1.0.in_layers.0.bias']
    unet_weights['decoder.1.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.1.0.in_layers.2.weight']
    unet_weights['decoder.1.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.1.0.in_layers.2.bias']
    unet_weights['decoder.1.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.1.0.emb_layers.1.weight']
    unet_weights['decoder.1.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.1.0.emb_layers.1.bias']
    unet_weights['decoder.1.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.1.0.out_layers.0.weight']
    unet_weights['decoder.1.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.1.0.out_layers.0.bias']
    unet_weights['decoder.1.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.1.0.out_layers.3.weight']
    unet_weights['decoder.1.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.1.0.out_layers.3.bias']
    unet_weights['decoder.1.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.1.0.skip_connection.weight']
    unet_weights['decoder.1.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.1.0.skip_connection.bias']     
                
    # SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),         # b 1280 h/32 w/32
    unet_weights['decoder.2.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.2.0.in_layers.0.weight']
    unet_weights['decoder.2.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.2.0.in_layers.0.bias']
    unet_weights['decoder.2.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.2.0.in_layers.2.weight']
    unet_weights['decoder.2.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.2.0.in_layers.2.bias']
    unet_weights['decoder.2.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.2.0.emb_layers.1.weight']
    unet_weights['decoder.2.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.2.0.emb_layers.1.bias']
    unet_weights['decoder.2.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.2.0.out_layers.0.weight']
    unet_weights['decoder.2.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.2.0.out_layers.0.bias']
    unet_weights['decoder.2.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.2.0.out_layers.3.weight']
    unet_weights['decoder.2.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.2.0.out_layers.3.bias']
    unet_weights['decoder.2.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.2.0.skip_connection.weight']
    unet_weights['decoder.2.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.2.0.skip_connection.bias']     
    unet_weights['decoder.2.1.conv.weight']                             = org_weights['model.diffusion_model.output_blocks.2.1.conv.weight']
    unet_weights['decoder.2.1.conv.bias']                               = org_weights['model.diffusion_model.output_blocks.2.1.conv.bias']

    # SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
    unet_weights['decoder.3.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.3.0.in_layers.0.weight']
    unet_weights['decoder.3.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.3.0.in_layers.0.bias']
    unet_weights['decoder.3.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.3.0.in_layers.2.weight']
    unet_weights['decoder.3.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.3.0.in_layers.2.bias']
    unet_weights['decoder.3.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.3.0.emb_layers.1.weight']
    unet_weights['decoder.3.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.3.0.emb_layers.1.bias']
    unet_weights['decoder.3.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.3.0.out_layers.0.weight']
    unet_weights['decoder.3.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.3.0.out_layers.0.bias']
    unet_weights['decoder.3.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.3.0.out_layers.3.weight']
    unet_weights['decoder.3.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.3.0.out_layers.3.bias']
    unet_weights['decoder.3.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.3.0.skip_connection.weight']
    unet_weights['decoder.3.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.3.0.skip_connection.bias']             
    unet_weights['decoder.3.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.3.1.norm.weight']
    unet_weights['decoder.3.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.3.1.norm.bias']
    unet_weights['decoder.3.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.3.1.proj_in.weight']
    unet_weights['decoder.3.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.3.1.proj_in.bias']
    unet_weights['decoder.3.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.3.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.3.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.3.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.3.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.3.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.3.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.3.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.3.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.3.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.3.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.3.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.3.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.3.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.3.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.3.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.3.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.3.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.3.1.proj_out.weight']
    unet_weights['decoder.3.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.3.1.proj_out.bias']
    unet_weights['decoder.3.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
    unet_weights['decoder.4.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.4.0.in_layers.0.weight']
    unet_weights['decoder.4.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.4.0.in_layers.0.bias']
    unet_weights['decoder.4.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.4.0.in_layers.2.weight']
    unet_weights['decoder.4.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.4.0.in_layers.2.bias']
    unet_weights['decoder.4.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.4.0.emb_layers.1.weight']
    unet_weights['decoder.4.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.4.0.emb_layers.1.bias']
    unet_weights['decoder.4.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.4.0.out_layers.0.weight']
    unet_weights['decoder.4.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.4.0.out_layers.0.bias']
    unet_weights['decoder.4.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.4.0.out_layers.3.weight']
    unet_weights['decoder.4.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.4.0.out_layers.3.bias']
    unet_weights['decoder.4.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.4.0.skip_connection.weight']
    unet_weights['decoder.4.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.4.0.skip_connection.bias']             
    unet_weights['decoder.4.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.4.1.norm.weight']
    unet_weights['decoder.4.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.4.1.norm.bias']
    unet_weights['decoder.4.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.4.1.proj_in.weight']
    unet_weights['decoder.4.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.4.1.proj_in.bias']
    unet_weights['decoder.4.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.4.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.4.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.4.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.4.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.4.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.4.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.4.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.4.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.4.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.4.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.4.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.4.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.4.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.4.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.4.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.4.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.4.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.4.1.proj_out.weight']
    unet_weights['decoder.4.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.4.1.proj_out.bias']
    unet_weights['decoder.4.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)


    # SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)), # b 1280 h/16 w/16
    unet_weights['decoder.5.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.5.0.in_layers.0.weight']
    unet_weights['decoder.5.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.5.0.in_layers.0.bias']
    unet_weights['decoder.5.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.5.0.in_layers.2.weight']
    unet_weights['decoder.5.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.5.0.in_layers.2.bias']
    unet_weights['decoder.5.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.5.0.emb_layers.1.weight']
    unet_weights['decoder.5.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.5.0.emb_layers.1.bias']
    unet_weights['decoder.5.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.5.0.out_layers.0.weight']
    unet_weights['decoder.5.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.5.0.out_layers.0.bias']
    unet_weights['decoder.5.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.5.0.out_layers.3.weight']
    unet_weights['decoder.5.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.5.0.out_layers.3.bias']
    unet_weights['decoder.5.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.5.0.skip_connection.weight']
    unet_weights['decoder.5.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.5.0.skip_connection.bias']             
    unet_weights['decoder.5.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.5.1.norm.weight']
    unet_weights['decoder.5.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.5.1.norm.bias']
    unet_weights['decoder.5.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.5.1.proj_in.weight']
    unet_weights['decoder.5.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.5.1.proj_in.bias']
    unet_weights['decoder.5.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.5.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.5.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.5.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.5.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.5.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.5.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.5.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.5.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.5.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.5.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.5.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.5.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.5.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.5.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.5.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.5.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.5.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.5.1.proj_out.weight']
    unet_weights['decoder.5.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.5.1.proj_out.bias']
    unet_weights['decoder.5.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)
    unet_weights['decoder.5.2.conv.weight']                             = org_weights['model.diffusion_model.output_blocks.5.2.conv.weight']
    unet_weights['decoder.5.2.conv.bias']                               = org_weights['model.diffusion_model.output_blocks.5.2.conv.bias']

    # SwitchSequential(ResidualBlock(1920, 640),  AttentionBlock(8, 80)),                  # b 640 h/16 w/16

    unet_weights['decoder.6.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.6.0.in_layers.0.weight']
    unet_weights['decoder.6.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.6.0.in_layers.0.bias']
    unet_weights['decoder.6.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.6.0.in_layers.2.weight']
    unet_weights['decoder.6.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.6.0.in_layers.2.bias']
    unet_weights['decoder.6.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.6.0.emb_layers.1.weight']
    unet_weights['decoder.6.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.6.0.emb_layers.1.bias']
    unet_weights['decoder.6.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.6.0.out_layers.0.weight']
    unet_weights['decoder.6.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.6.0.out_layers.0.bias']
    unet_weights['decoder.6.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.6.0.out_layers.3.weight']
    unet_weights['decoder.6.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.6.0.out_layers.3.bias']
    unet_weights['decoder.6.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.6.0.skip_connection.weight']
    unet_weights['decoder.6.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.6.0.skip_connection.bias']             
    unet_weights['decoder.6.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.6.1.norm.weight']
    unet_weights['decoder.6.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.6.1.norm.bias']
    unet_weights['decoder.6.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.6.1.proj_in.weight']
    unet_weights['decoder.6.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.6.1.proj_in.bias']
    unet_weights['decoder.6.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.6.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.6.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.6.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.6.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.6.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.6.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.6.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.6.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.6.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.6.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.6.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.6.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.6.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.6.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.6.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.6.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.6.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.6.1.proj_out.weight']
    unet_weights['decoder.6.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.6.1.proj_out.bias']
    unet_weights['decoder.6.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(ResidualBlock(1280, 640),  AttentionBlock(8, 80)),                
    unet_weights['decoder.7.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.7.0.in_layers.0.weight']
    unet_weights['decoder.7.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.7.0.in_layers.0.bias']
    unet_weights['decoder.7.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.7.0.in_layers.2.weight']
    unet_weights['decoder.7.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.7.0.in_layers.2.bias']
    unet_weights['decoder.7.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.7.0.emb_layers.1.weight']
    unet_weights['decoder.7.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.7.0.emb_layers.1.bias']
    unet_weights['decoder.7.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.7.0.out_layers.0.weight']
    unet_weights['decoder.7.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.7.0.out_layers.0.bias']
    unet_weights['decoder.7.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.7.0.out_layers.3.weight']
    unet_weights['decoder.7.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.7.0.out_layers.3.bias']
    unet_weights['decoder.7.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.7.0.skip_connection.weight']
    unet_weights['decoder.7.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.7.0.skip_connection.bias']             
    unet_weights['decoder.7.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.7.1.norm.weight']
    unet_weights['decoder.7.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.7.1.norm.bias']
    unet_weights['decoder.7.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.7.1.proj_in.weight']
    unet_weights['decoder.7.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.7.1.proj_in.bias']
    unet_weights['decoder.7.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.7.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.7.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.7.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.7.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.7.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.7.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.7.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.7.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.7.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.7.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.7.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.7.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.7.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.7.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.7.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.7.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.7.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.7.1.proj_out.weight']
    unet_weights['decoder.7.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.7.1.proj_out.bias']
    unet_weights['decoder.7.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)
    # SwitchSequential(ResidualBlock(960, 640),   AttentionBlock(8, 80), Upsample(640)), # b 640 h/8 w/8
    unet_weights['decoder.8.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.8.0.in_layers.0.weight']
    unet_weights['decoder.8.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.8.0.in_layers.0.bias']
    unet_weights['decoder.8.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.8.0.in_layers.2.weight']
    unet_weights['decoder.8.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.8.0.in_layers.2.bias']
    unet_weights['decoder.8.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.8.0.emb_layers.1.weight']
    unet_weights['decoder.8.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.8.0.emb_layers.1.bias']
    unet_weights['decoder.8.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.8.0.out_layers.0.weight']
    unet_weights['decoder.8.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.8.0.out_layers.0.bias']
    unet_weights['decoder.8.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.8.0.out_layers.3.weight']
    unet_weights['decoder.8.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.8.0.out_layers.3.bias']
    unet_weights['decoder.8.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.8.0.skip_connection.weight']
    unet_weights['decoder.8.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.8.0.skip_connection.bias']             
    unet_weights['decoder.8.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.8.1.norm.weight']
    unet_weights['decoder.8.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.8.1.norm.bias']
    unet_weights['decoder.8.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.8.1.proj_in.weight']
    unet_weights['decoder.8.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.8.1.proj_in.bias']
    unet_weights['decoder.8.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.8.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.8.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.8.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.8.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.8.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.8.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.8.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.8.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.8.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.8.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.8.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.8.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.8.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.8.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.8.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.8.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.8.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.8.1.proj_out.weight']
    unet_weights['decoder.8.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.8.1.proj_out.bias']
    unet_weights['decoder.8.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)
    unet_weights['decoder.8.2.conv.weight']                             = org_weights['model.diffusion_model.output_blocks.8.2.conv.weight']
    unet_weights['decoder.8.2.conv.bias']                               = org_weights['model.diffusion_model.output_blocks.8.2.conv.bias']

    # SwitchSequential(ResidualBlock(960, 320),   AttentionBlock(8, 40)),                # b 320 h/8 w/8
    unet_weights['decoder.9.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.9.0.in_layers.0.weight']
    unet_weights['decoder.9.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.9.0.in_layers.0.bias']
    unet_weights['decoder.9.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.9.0.in_layers.2.weight']
    unet_weights['decoder.9.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.9.0.in_layers.2.bias']
    unet_weights['decoder.9.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.9.0.emb_layers.1.weight']
    unet_weights['decoder.9.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.9.0.emb_layers.1.bias']
    unet_weights['decoder.9.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.9.0.out_layers.0.weight']
    unet_weights['decoder.9.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.9.0.out_layers.0.bias']
    unet_weights['decoder.9.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.9.0.out_layers.3.weight']
    unet_weights['decoder.9.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.9.0.out_layers.3.bias']
    unet_weights['decoder.9.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.9.0.skip_connection.weight']
    unet_weights['decoder.9.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.9.0.skip_connection.bias']             
    unet_weights['decoder.9.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.9.1.norm.weight']
    unet_weights['decoder.9.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.9.1.norm.bias']
    unet_weights['decoder.9.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.9.1.proj_in.weight']
    unet_weights['decoder.9.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.9.1.proj_in.bias']
    unet_weights['decoder.9.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.9.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.9.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.9.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.9.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.9.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.9.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.9.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.9.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.9.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.9.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.9.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.9.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.9.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.9.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.9.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.9.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.9.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.9.1.proj_out.weight']
    unet_weights['decoder.9.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.9.1.proj_out.bias']
    unet_weights['decoder.9.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    # SwitchSequential(ResidualBlock(640, 320),   AttentionBlock(8, 40)),
    unet_weights['decoder.10.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.10.0.in_layers.0.weight']
    unet_weights['decoder.10.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.10.0.in_layers.0.bias']
    unet_weights['decoder.10.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.10.0.in_layers.2.weight']
    unet_weights['decoder.10.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.10.0.in_layers.2.bias']
    unet_weights['decoder.10.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.10.0.emb_layers.1.weight']
    unet_weights['decoder.10.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.10.0.emb_layers.1.bias']
    unet_weights['decoder.10.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.10.0.out_layers.0.weight']
    unet_weights['decoder.10.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.10.0.out_layers.0.bias']
    unet_weights['decoder.10.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.10.0.out_layers.3.weight']
    unet_weights['decoder.10.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.10.0.out_layers.3.bias']
    unet_weights['decoder.10.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.10.0.skip_connection.weight']
    unet_weights['decoder.10.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.10.0.skip_connection.bias']             
    unet_weights['decoder.10.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.10.1.norm.weight']
    unet_weights['decoder.10.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.10.1.norm.bias']
    unet_weights['decoder.10.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.10.1.proj_in.weight']
    unet_weights['decoder.10.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.10.1.proj_in.bias']
    unet_weights['decoder.10.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.10.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.10.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.10.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.10.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.10.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.10.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.10.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.10.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.10.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.10.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.10.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.10.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.10.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.10.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.10.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.10.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.10.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.10.1.proj_out.weight']
    unet_weights['decoder.10.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.10.1.proj_out.bias']
    unet_weights['decoder.10.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)


    # SwitchSequential(ResidualBlock(640, 320),   AttentionBlock(8, 40))
    unet_weights['decoder.11.0.x_trans.0.weight']                        = org_weights['model.diffusion_model.output_blocks.11.0.in_layers.0.weight']
    unet_weights['decoder.11.0.x_trans.0.bias']                          = org_weights['model.diffusion_model.output_blocks.11.0.in_layers.0.bias']
    unet_weights['decoder.11.0.x_trans.2.weight']                        = org_weights['model.diffusion_model.output_blocks.11.0.in_layers.2.weight']
    unet_weights['decoder.11.0.x_trans.2.bias']                          = org_weights['model.diffusion_model.output_blocks.11.0.in_layers.2.bias']
    unet_weights['decoder.11.0.t_trans.1.weight']                        = org_weights['model.diffusion_model.output_blocks.11.0.emb_layers.1.weight']
    unet_weights['decoder.11.0.t_trans.1.bias']                          = org_weights['model.diffusion_model.output_blocks.11.0.emb_layers.1.bias']
    unet_weights['decoder.11.0.out_trans.0.weight']                      = org_weights['model.diffusion_model.output_blocks.11.0.out_layers.0.weight']
    unet_weights['decoder.11.0.out_trans.0.bias']                        = org_weights['model.diffusion_model.output_blocks.11.0.out_layers.0.bias']
    unet_weights['decoder.11.0.out_trans.2.weight']                      = org_weights['model.diffusion_model.output_blocks.11.0.out_layers.3.weight']
    unet_weights['decoder.11.0.out_trans.2.bias']                        = org_weights['model.diffusion_model.output_blocks.11.0.out_layers.3.bias']
    unet_weights['decoder.11.0.resize.weight']                           = org_weights['model.diffusion_model.output_blocks.11.0.skip_connection.weight']
    unet_weights['decoder.11.0.resize.bias']                             = org_weights['model.diffusion_model.output_blocks.11.0.skip_connection.bias']             
    unet_weights['decoder.11.1.norm_conv1.0.weight']                     = org_weights['model.diffusion_model.output_blocks.11.1.norm.weight']
    unet_weights['decoder.11.1.norm_conv1.0.bias']                       = org_weights['model.diffusion_model.output_blocks.11.1.norm.bias']
    unet_weights['decoder.11.1.norm_conv1.1.weight']                     = org_weights['model.diffusion_model.output_blocks.11.1.proj_in.weight']
    unet_weights['decoder.11.1.norm_conv1.1.bias']                       = org_weights['model.diffusion_model.output_blocks.11.1.proj_in.bias']
    unet_weights['decoder.11.1.norm_self_att.0.weight']                  = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.weight']
    unet_weights['decoder.11.1.norm_self_att.0.bias']                    = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias']
    unet_weights['decoder.11.1.norm_self_att.1.w_o.weight']              = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.weight']
    unet_weights['decoder.11.1.norm_self_att.1.w_o.bias']                = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.bias']
    unet_weights['decoder.11.1.norm.weight']                             = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.weight']
    unet_weights['decoder.11.1.norm.bias']                               = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.bias']
    unet_weights['decoder.11.1.cross_att.w_q.weight']                    = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight']
    unet_weights['decoder.11.1.cross_att.w_k.weight']                    = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight']
    unet_weights['decoder.11.1.cross_att.w_v.weight']                    = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight']
    unet_weights['decoder.11.1.cross_att.w_o.weight']                    = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight']
    unet_weights['decoder.11.1.cross_att.w_o.bias']                      = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias']
    unet_weights['decoder.11.1.geglu.0.weight']                          = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.weight']
    unet_weights['decoder.11.1.geglu.0.bias']                            = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.bias']
    unet_weights['decoder.11.1.geglu.1.fc1.weight']                      = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.weight']
    unet_weights['decoder.11.1.geglu.1.fc1.bias']                        = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.bias']
    unet_weights['decoder.11.1.geglu.1.fc2.weight']                      = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.weight']
    unet_weights['decoder.11.1.geglu.1.fc2.bias']                        = org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.bias']
    unet_weights['decoder.11.1.conv_out.weight']                         = org_weights['model.diffusion_model.output_blocks.11.1.proj_out.weight']
    unet_weights['decoder.11.1.conv_out.bias']                           = org_weights['model.diffusion_model.output_blocks.11.1.proj_out.bias']
    unet_weights['decoder.11.1.norm_self_att.1.w_qkv.weight'] = torch.cat((org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight'], 
                                                                        org_weights['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.weight']), dim=0)

    unet_weights['out.0.weight'] = org_weights['model.diffusion_model.out.0.weight']
    unet_weights['out.0.bias']   = org_weights['model.diffusion_model.out.0.bias']
    unet_weights['out.2.weight'] = org_weights['model.diffusion_model.out.2.weight'] 
    unet_weights['out.2.bias']   = org_weights['model.diffusion_model.out.2.bias']
    return unet_weights

def get_clip_weights(org_weights):
    clip_weights = {}

    clip_weights['e.p_emb']        = org_weights['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight'] 
    clip_weights['e.t_emb.weight'] = org_weights['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
    clip_weights['norm.weight']    = org_weights['cond_stage_model.transformer.text_model.final_layer_norm.weight']
    clip_weights['norm.bias']      = org_weights['cond_stage_model.transformer.text_model.final_layer_norm.bias']
    for i in range(12):
        clip_weights[f"layers.{i}.norm1.weight"]  = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.weight']
        clip_weights[f"layers.{i}.norm1.bias"]    = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.bias']    
        clip_weights[f"layers.{i}.norm2.weight"]  = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.weight']
        clip_weights[f"layers.{i}.norm2.bias"]    = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.bias']

        clip_weights[f"layers.{i}.attn.w_o.weight"] = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.out_proj.weight']
        clip_weights[f"layers.{i}.attn.w_o.bias"]   = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.out_proj.bias']
        clip_weights[f"layers.{i}.fc1.weight"]      = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.weight']
        clip_weights[f"layers.{i}.fc1.bias"]        = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.bias']
        clip_weights[f"layers.{i}.fc2.weight"]      = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.weight']
        clip_weights[f"layers.{i}.fc2.bias"]        = org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.bias']

        clip_weights[f"layers.{i}.attn.w_qkv.bias"] = torch.cat([org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.bias'],
                                                            org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.bias'],
                                                            org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.v_proj.bias']], dim=0)
        
        clip_weights[f"layers.{i}.attn.w_qkv.weight"] = torch.cat([org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.weight'],
                                                            org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.weight'],
                                                            org_weights[f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.v_proj.weight']], dim=0)


    return clip_weights

def get_vae_encoder_weights(org_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    encoder_weights = {}

    encoder_weights['layers.0.weight']       =  org_weights['first_stage_model.encoder.conv_in.weight']
    encoder_weights['layers.0.bias']         =  org_weights['first_stage_model.encoder.conv_in.bias']

    encoder_weights['layers.1.norm1.weight'] =  org_weights['first_stage_model.encoder.down.0.block.0.norm1.weight']
    encoder_weights['layers.1.norm1.bias']   =  org_weights['first_stage_model.encoder.down.0.block.0.norm1.bias']
    encoder_weights['layers.1.norm2.weight'] =  org_weights['first_stage_model.encoder.down.0.block.0.norm2.weight']
    encoder_weights['layers.1.norm2.bias']   =  org_weights['first_stage_model.encoder.down.0.block.0.norm2.bias']

    encoder_weights['layers.1.conv1.weight'] =  org_weights['first_stage_model.encoder.down.0.block.0.conv1.weight']
    encoder_weights['layers.1.conv1.bias']   =  org_weights['first_stage_model.encoder.down.0.block.0.conv1.bias']
    encoder_weights['layers.1.conv2.weight'] =  org_weights['first_stage_model.encoder.down.0.block.0.conv2.weight'] 
    encoder_weights['layers.1.conv2.bias']   =  org_weights['first_stage_model.encoder.down.0.block.0.conv2.bias']

    encoder_weights['layers.2.norm1.weight'] =  org_weights['first_stage_model.encoder.down.0.block.1.norm1.weight']
    encoder_weights['layers.2.norm1.bias']   =  org_weights['first_stage_model.encoder.down.0.block.1.norm1.bias']
    encoder_weights['layers.2.norm2.weight'] =  org_weights['first_stage_model.encoder.down.0.block.1.norm2.weight']
    encoder_weights['layers.2.norm2.bias']   =  org_weights['first_stage_model.encoder.down.0.block.1.norm2.bias']

    encoder_weights['layers.2.conv1.weight'] =  org_weights['first_stage_model.encoder.down.0.block.1.conv1.weight']
    encoder_weights['layers.2.conv1.bias']   =  org_weights['first_stage_model.encoder.down.0.block.1.conv1.bias']
    encoder_weights['layers.2.conv2.weight'] =  org_weights['first_stage_model.encoder.down.0.block.1.conv2.weight']
    encoder_weights['layers.2.conv2.bias']   =  org_weights['first_stage_model.encoder.down.0.block.1.conv2.bias']

    encoder_weights['layers.3.weight']       =  org_weights['first_stage_model.encoder.down.0.downsample.conv.weight']
    encoder_weights['layers.3.bias']         =  org_weights['first_stage_model.encoder.down.0.downsample.conv.bias']
                    

    encoder_weights['layers.4.norm1.weight'] = org_weights['first_stage_model.encoder.down.1.block.0.norm1.weight']
    encoder_weights['layers.4.norm1.bias']   = org_weights['first_stage_model.encoder.down.1.block.0.norm1.bias']
    encoder_weights['layers.4.norm2.weight'] = org_weights['first_stage_model.encoder.down.1.block.0.norm2.weight']
    encoder_weights['layers.4.norm2.bias']   = org_weights['first_stage_model.encoder.down.1.block.0.norm2.bias']

    encoder_weights['layers.4.conv1.weight'] = org_weights['first_stage_model.encoder.down.1.block.0.conv1.weight']
    encoder_weights['layers.4.conv1.bias']   = org_weights['first_stage_model.encoder.down.1.block.0.conv1.bias']
    encoder_weights['layers.4.conv2.weight'] = org_weights['first_stage_model.encoder.down.1.block.0.conv2.weight']
    encoder_weights['layers.4.conv2.bias']   = org_weights['first_stage_model.encoder.down.1.block.0.conv2.bias']

    encoder_weights['layers.4.resize.weight']= org_weights['first_stage_model.encoder.down.1.block.0.nin_shortcut.weight']
    encoder_weights['layers.4.resize.bias']  = org_weights['first_stage_model.encoder.down.1.block.0.nin_shortcut.bias']


    encoder_weights['layers.5.norm1.weight'] = org_weights['first_stage_model.encoder.down.1.block.1.norm1.weight']
    encoder_weights['layers.5.norm1.bias']   = org_weights['first_stage_model.encoder.down.1.block.1.norm1.bias']
    encoder_weights['layers.5.norm2.weight'] = org_weights['first_stage_model.encoder.down.1.block.1.norm2.weight']
    encoder_weights['layers.5.norm2.bias']   = org_weights['first_stage_model.encoder.down.1.block.1.norm2.bias']
                    
    encoder_weights['layers.5.conv1.weight'] = org_weights['first_stage_model.encoder.down.1.block.1.conv1.weight']
    encoder_weights['layers.5.conv1.bias']   = org_weights['first_stage_model.encoder.down.1.block.1.conv1.bias']
    encoder_weights['layers.5.conv2.weight'] = org_weights['first_stage_model.encoder.down.1.block.1.conv2.weight']
    encoder_weights['layers.5.conv2.bias']   = org_weights['first_stage_model.encoder.down.1.block.1.conv2.bias']
                    
    encoder_weights['layers.6.weight']       = org_weights['first_stage_model.encoder.down.1.downsample.conv.weight']
    encoder_weights['layers.6.bias']         = org_weights['first_stage_model.encoder.down.1.downsample.conv.bias']
                    
    encoder_weights['layers.7.norm1.weight'] = org_weights['first_stage_model.encoder.down.2.block.0.norm1.weight']
    encoder_weights['layers.7.norm1.bias']   = org_weights['first_stage_model.encoder.down.2.block.0.norm1.bias']
    encoder_weights['layers.7.norm2.weight'] = org_weights['first_stage_model.encoder.down.2.block.0.norm2.weight']
    encoder_weights['layers.7.norm2.bias']   = org_weights['first_stage_model.encoder.down.2.block.0.norm2.bias']
                    
    encoder_weights['layers.7.conv1.weight'] = org_weights['first_stage_model.encoder.down.2.block.0.conv1.weight']
    encoder_weights['layers.7.conv1.bias']   = org_weights['first_stage_model.encoder.down.2.block.0.conv1.bias']
    encoder_weights['layers.7.conv2.weight'] = org_weights['first_stage_model.encoder.down.2.block.0.conv2.weight']
    encoder_weights['layers.7.conv2.bias']   = org_weights['first_stage_model.encoder.down.2.block.0.conv2.bias']
                    
    encoder_weights['layers.7.resize.weight'] = org_weights['first_stage_model.encoder.down.2.block.0.nin_shortcut.weight']
    encoder_weights['layers.7.resize.bias']   = org_weights['first_stage_model.encoder.down.2.block.0.nin_shortcut.bias']
                    
    encoder_weights['layers.8.norm1.weight']  = org_weights['first_stage_model.encoder.down.2.block.1.norm1.weight']
    encoder_weights['layers.8.norm1.bias']    = org_weights['first_stage_model.encoder.down.2.block.1.norm1.bias']
    encoder_weights['layers.8.norm2.weight']  = org_weights['first_stage_model.encoder.down.2.block.1.norm2.weight']
    encoder_weights['layers.8.norm2.bias']    = org_weights['first_stage_model.encoder.down.2.block.1.norm2.bias']
                    
    encoder_weights['layers.8.conv1.weight']  = org_weights['first_stage_model.encoder.down.2.block.1.conv1.weight']
    encoder_weights['layers.8.conv1.bias']    = org_weights['first_stage_model.encoder.down.2.block.1.conv1.bias']
    encoder_weights['layers.8.conv2.weight']  = org_weights['first_stage_model.encoder.down.2.block.1.conv2.weight']
    encoder_weights['layers.8.conv2.bias']    = org_weights['first_stage_model.encoder.down.2.block.1.conv2.bias']

    encoder_weights['layers.9.weight']        = org_weights['first_stage_model.encoder.down.2.downsample.conv.weight']
    encoder_weights['layers.9.bias']          = org_weights['first_stage_model.encoder.down.2.downsample.conv.bias']


    encoder_weights['layers.10.norm1.weight']        = org_weights['first_stage_model.encoder.down.3.block.0.norm1.weight']
    encoder_weights['layers.10.norm1.bias']          = org_weights['first_stage_model.encoder.down.3.block.0.norm1.bias']
    encoder_weights['layers.10.norm2.weight']        = org_weights['first_stage_model.encoder.down.3.block.0.norm2.weight']
    encoder_weights['layers.10.norm2.bias']          = org_weights['first_stage_model.encoder.down.3.block.0.norm2.bias']
                    
    encoder_weights['layers.10.conv1.weight']        = org_weights['first_stage_model.encoder.down.3.block.0.conv1.weight']
    encoder_weights['layers.10.conv1.bias']          = org_weights['first_stage_model.encoder.down.3.block.0.conv1.bias']
    encoder_weights['layers.10.conv2.weight']        = org_weights['first_stage_model.encoder.down.3.block.0.conv2.weight']
    encoder_weights['layers.10.conv2.bias']          = org_weights['first_stage_model.encoder.down.3.block.0.conv2.bias']
                    
    encoder_weights['layers.11.norm1.weight']        = org_weights['first_stage_model.encoder.down.3.block.1.norm1.weight']
    encoder_weights['layers.11.norm1.bias']          = org_weights['first_stage_model.encoder.down.3.block.1.norm1.bias']
    encoder_weights['layers.11.norm2.weight']        = org_weights['first_stage_model.encoder.down.3.block.1.norm2.weight']
    encoder_weights['layers.11.norm2.bias']          = org_weights['first_stage_model.encoder.down.3.block.1.norm2.bias']
                    
    encoder_weights['layers.11.conv1.weight']        = org_weights['first_stage_model.encoder.down.3.block.1.conv1.weight']
    encoder_weights['layers.11.conv1.bias']          = org_weights['first_stage_model.encoder.down.3.block.1.conv1.bias']
    encoder_weights['layers.11.conv2.weight']        = org_weights['first_stage_model.encoder.down.3.block.1.conv2.weight']
    encoder_weights['layers.11.conv2.bias']          = org_weights['first_stage_model.encoder.down.3.block.1.conv2.bias']
                    
    encoder_weights['layers.12.norm1.weight']        = org_weights['first_stage_model.encoder.mid.block_1.norm1.weight']
    encoder_weights['layers.12.norm1.bias']         = org_weights['first_stage_model.encoder.mid.block_1.norm1.bias']
    encoder_weights['layers.12.norm2.weight']        = org_weights['first_stage_model.encoder.mid.block_1.norm2.weight']
    encoder_weights['layers.12.norm2.bias']         = org_weights['first_stage_model.encoder.mid.block_1.norm2.bias']

    encoder_weights['layers.12.conv1.weight']        = org_weights['first_stage_model.encoder.mid.block_1.conv1.weight']
    encoder_weights['layers.12.conv1.bias']         = org_weights['first_stage_model.encoder.mid.block_1.conv1.bias']
    encoder_weights['layers.12.conv2.weight']        = org_weights['first_stage_model.encoder.mid.block_1.conv2.weight']
    encoder_weights['layers.12.conv2.bias']         = org_weights['first_stage_model.encoder.mid.block_1.conv2.bias']
                    
    encoder_weights['layers.13.norm.weight']       = org_weights['first_stage_model.encoder.mid.attn_1.norm.weight']
    encoder_weights['layers.13.norm.bias']         = org_weights['first_stage_model.encoder.mid.attn_1.norm.bias']
                    
    encoder_weights['layers.14.norm1.weight']       = org_weights['first_stage_model.encoder.mid.block_2.norm1.weight']
    encoder_weights['layers.14.norm1.bias']         = org_weights['first_stage_model.encoder.mid.block_2.norm1.bias']
    encoder_weights['layers.14.norm2.weight']       = org_weights['first_stage_model.encoder.mid.block_2.norm2.weight']
    encoder_weights['layers.14.norm2.bias']         = org_weights['first_stage_model.encoder.mid.block_2.norm2.bias']
                    
    encoder_weights['layers.14.conv1.weight']       = org_weights['first_stage_model.encoder.mid.block_2.conv1.weight']
    encoder_weights['layers.14.conv1.bias']         = org_weights['first_stage_model.encoder.mid.block_2.conv1.bias']
    encoder_weights['layers.14.conv2.weight']       = org_weights['first_stage_model.encoder.mid.block_2.conv2.weight']
    encoder_weights['layers.14.conv2.bias']         = org_weights['first_stage_model.encoder.mid.block_2.conv2.bias']
                    
    encoder_weights['layers.15.weight']         = org_weights['first_stage_model.encoder.norm_out.weight']
    encoder_weights['layers.15.bias']           = org_weights['first_stage_model.encoder.norm_out.bias']
                
    encoder_weights['layers.17.weight']         = org_weights['first_stage_model.encoder.conv_out.weight']
    encoder_weights['layers.17.bias']           = org_weights['first_stage_model.encoder.conv_out.bias']
                    
    encoder_weights['layers.18.weight']         = org_weights['first_stage_model.quant_conv.weight']  
    encoder_weights['layers.18.bias']           = org_weights['first_stage_model.quant_conv.bias']

    encoder_weights['layers.13.attn.w_o.weight'] = org_weights['first_stage_model.encoder.mid.attn_1.proj_out.weight'].squeeze()
    encoder_weights['layers.13.attn.w_o.bias']   = org_weights['first_stage_model.encoder.mid.attn_1.proj_out.bias']

    encoder_weights['layers.13.attn.w_qkv.bias'] = torch.cat([org_weights['first_stage_model.encoder.mid.attn_1.q.bias'], 
                                                            org_weights['first_stage_model.encoder.mid.attn_1.k.bias'],
                                                            org_weights['first_stage_model.encoder.mid.attn_1.v.bias']], dim=0)

    encoder_weights['layers.13.attn.w_qkv.weight'] = torch.cat([org_weights['first_stage_model.encoder.mid.attn_1.q.weight'], 
                                                                org_weights['first_stage_model.encoder.mid.attn_1.k.weight'],
                                                                org_weights['first_stage_model.encoder.mid.attn_1.v.weight']], dim=0).squeeze()

    return encoder_weights

def get_vae_decoder_weights(org_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    decoder_weights = {}

    decoder_weights['0.weight'] = org_weights['first_stage_model.post_quant_conv.weight']
    decoder_weights['0.bias']   = org_weights['first_stage_model.post_quant_conv.bias']
    decoder_weights['1.weight'] = org_weights['first_stage_model.decoder.conv_in.weight'] 
    decoder_weights['1.bias']   = org_weights['first_stage_model.decoder.conv_in.bias']

    decoder_weights['2.norm1.weight'] = org_weights['first_stage_model.decoder.mid.block_1.norm1.weight']
    decoder_weights['2.norm1.bias']   = org_weights['first_stage_model.decoder.mid.block_1.norm1.bias']
    decoder_weights['2.norm2.weight'] = org_weights['first_stage_model.decoder.mid.block_1.norm2.weight']
    decoder_weights['2.norm2.bias']   = org_weights['first_stage_model.decoder.mid.block_1.norm2.bias']
                    
    decoder_weights['2.conv1.weight'] = org_weights['first_stage_model.decoder.mid.block_1.conv1.weight']
    decoder_weights['2.conv1.bias']   = org_weights['first_stage_model.decoder.mid.block_1.conv1.bias']
    decoder_weights['2.conv2.weight'] = org_weights['first_stage_model.decoder.mid.block_1.conv2.weight']
    decoder_weights['2.conv2.bias']   = org_weights['first_stage_model.decoder.mid.block_1.conv2.bias']

    decoder_weights['3.norm.weight'] = org_weights['first_stage_model.decoder.mid.attn_1.norm.weight']
    decoder_weights['3.norm.bias']   = org_weights['first_stage_model.decoder.mid.attn_1.norm.bias']
                    
                    
    decoder_weights['3.attn.w_o.weight'] = org_weights['first_stage_model.decoder.mid.attn_1.proj_out.weight'].squeeze()
    decoder_weights['3.attn.w_o.bias']   = org_weights['first_stage_model.decoder.mid.attn_1.proj_out.bias']
                    
    decoder_weights['4.norm1.weight']   = org_weights['first_stage_model.decoder.mid.block_2.norm1.weight']
    decoder_weights['4.norm1.bias']     = org_weights['first_stage_model.decoder.mid.block_2.norm1.bias']
    decoder_weights['4.norm2.weight']   = org_weights['first_stage_model.decoder.mid.block_2.norm2.weight']
    decoder_weights['4.norm2.bias']     = org_weights['first_stage_model.decoder.mid.block_2.norm2.bias']
                    
    decoder_weights['4.conv1.weight']   = org_weights['first_stage_model.decoder.mid.block_2.conv1.weight']
    decoder_weights['4.conv1.bias']     = org_weights['first_stage_model.decoder.mid.block_2.conv1.bias']
    decoder_weights['4.conv2.weight']   = org_weights['first_stage_model.decoder.mid.block_2.conv2.weight']
    decoder_weights['4.conv2.bias']     = org_weights['first_stage_model.decoder.mid.block_2.conv2.bias']
                    
    decoder_weights['5.norm1.weight']     = org_weights['first_stage_model.decoder.up.3.block.0.norm1.weight']
    decoder_weights['5.norm1.bias']     = org_weights['first_stage_model.decoder.up.3.block.0.norm1.bias']
    decoder_weights['5.norm2.weight']     = org_weights['first_stage_model.decoder.up.3.block.0.norm2.weight']
    decoder_weights['5.norm2.bias']     = org_weights['first_stage_model.decoder.up.3.block.0.norm2.bias']
    decoder_weights['5.conv1.weight']     = org_weights['first_stage_model.decoder.up.3.block.0.conv1.weight']
    decoder_weights['5.conv1.bias']     = org_weights['first_stage_model.decoder.up.3.block.0.conv1.bias']
    decoder_weights['5.conv2.weight']     = org_weights['first_stage_model.decoder.up.3.block.0.conv2.weight']
    decoder_weights['5.conv2.bias']     = org_weights['first_stage_model.decoder.up.3.block.0.conv2.bias']
                    
    decoder_weights['6.norm1.weight']     = org_weights['first_stage_model.decoder.up.3.block.1.norm1.weight']
    decoder_weights['6.norm1.bias']     = org_weights['first_stage_model.decoder.up.3.block.1.norm1.bias']
    decoder_weights['6.norm2.weight']     = org_weights['first_stage_model.decoder.up.3.block.1.norm2.weight']
    decoder_weights['6.norm2.bias']     = org_weights['first_stage_model.decoder.up.3.block.1.norm2.bias']
    decoder_weights['6.conv1.weight']     = org_weights['first_stage_model.decoder.up.3.block.1.conv1.weight']
    decoder_weights['6.conv1.bias']     = org_weights['first_stage_model.decoder.up.3.block.1.conv1.bias']
    decoder_weights['6.conv2.weight']     = org_weights['first_stage_model.decoder.up.3.block.1.conv2.weight']
    decoder_weights['6.conv2.bias']     = org_weights['first_stage_model.decoder.up.3.block.1.conv2.bias']
                    
    decoder_weights['7.norm1.weight']     = org_weights['first_stage_model.decoder.up.3.block.2.norm1.weight']
    decoder_weights['7.norm1.bias']     = org_weights['first_stage_model.decoder.up.3.block.2.norm1.bias']
    decoder_weights['7.norm2.weight']     = org_weights['first_stage_model.decoder.up.3.block.2.norm2.weight']
    decoder_weights['7.norm2.bias']     = org_weights['first_stage_model.decoder.up.3.block.2.norm2.bias']
    decoder_weights['7.conv1.weight']     = org_weights['first_stage_model.decoder.up.3.block.2.conv1.weight']
    decoder_weights['7.conv1.bias']     = org_weights['first_stage_model.decoder.up.3.block.2.conv1.bias']
    decoder_weights['7.conv2.weight']     = org_weights['first_stage_model.decoder.up.3.block.2.conv2.weight']
    decoder_weights['7.conv2.bias']     = org_weights['first_stage_model.decoder.up.3.block.2.conv2.bias']
                    
    decoder_weights['9.weight']     = org_weights['first_stage_model.decoder.up.3.upsample.conv.weight']
    decoder_weights['9.bias']       = org_weights['first_stage_model.decoder.up.3.upsample.conv.bias']
                    
    decoder_weights['10.norm1.weight']     = org_weights['first_stage_model.decoder.up.2.block.0.norm1.weight']
    decoder_weights['10.norm1.bias']     = org_weights['first_stage_model.decoder.up.2.block.0.norm1.bias']
    decoder_weights['10.norm2.weight']     = org_weights['first_stage_model.decoder.up.2.block.0.norm2.weight']
    decoder_weights['10.norm2.bias']     = org_weights['first_stage_model.decoder.up.2.block.0.norm2.bias']
    decoder_weights['10.conv1.weight']     = org_weights['first_stage_model.decoder.up.2.block.0.conv1.weight']
    decoder_weights['10.conv1.bias']     = org_weights['first_stage_model.decoder.up.2.block.0.conv1.bias']
    decoder_weights['10.conv2.weight']     = org_weights['first_stage_model.decoder.up.2.block.0.conv2.weight']
    decoder_weights['10.conv2.bias']     = org_weights['first_stage_model.decoder.up.2.block.0.conv2.bias']
                    
    decoder_weights['11.norm1.weight']     = org_weights['first_stage_model.decoder.up.2.block.1.norm1.weight']
    decoder_weights['11.norm1.bias']       = org_weights['first_stage_model.decoder.up.2.block.1.norm1.bias']
    decoder_weights['11.norm2.weight']     = org_weights['first_stage_model.decoder.up.2.block.1.norm2.weight']
    decoder_weights['11.norm2.bias']       = org_weights['first_stage_model.decoder.up.2.block.1.norm2.bias']
    decoder_weights['11.conv1.weight']     = org_weights['first_stage_model.decoder.up.2.block.1.conv1.weight']
    decoder_weights['11.conv1.bias']       = org_weights['first_stage_model.decoder.up.2.block.1.conv1.bias']
    decoder_weights['11.conv2.weight']     = org_weights['first_stage_model.decoder.up.2.block.1.conv2.weight']
    decoder_weights['11.conv2.bias']       = org_weights['first_stage_model.decoder.up.2.block.1.conv2.bias']
                    
    decoder_weights['12.norm1.weight']     = org_weights['first_stage_model.decoder.up.2.block.2.norm1.weight']
    decoder_weights['12.norm1.bias']       = org_weights['first_stage_model.decoder.up.2.block.2.norm1.bias']
    decoder_weights['12.norm2.weight']     = org_weights['first_stage_model.decoder.up.2.block.2.norm2.weight']
    decoder_weights['12.norm2.bias']       = org_weights['first_stage_model.decoder.up.2.block.2.norm2.bias']
    decoder_weights['12.conv1.weight']     = org_weights['first_stage_model.decoder.up.2.block.2.conv1.weight']
    decoder_weights['12.conv1.bias']       = org_weights['first_stage_model.decoder.up.2.block.2.conv1.bias']
    decoder_weights['12.conv2.weight']     = org_weights['first_stage_model.decoder.up.2.block.2.conv2.weight']
    decoder_weights['12.conv2.bias']       = org_weights['first_stage_model.decoder.up.2.block.2.conv2.bias']
                    
    decoder_weights['14.weight']     = org_weights['first_stage_model.decoder.up.2.upsample.conv.weight']
    decoder_weights['14.bias']       = org_weights['first_stage_model.decoder.up.2.upsample.conv.bias']
                    

    decoder_weights['15.norm1.weight']     = org_weights['first_stage_model.decoder.up.1.block.0.norm1.weight']
    decoder_weights['15.norm1.bias']       = org_weights['first_stage_model.decoder.up.1.block.0.norm1.bias']
    decoder_weights['15.norm2.weight']     = org_weights['first_stage_model.decoder.up.1.block.0.norm2.weight']
    decoder_weights['15.norm2.bias']       = org_weights['first_stage_model.decoder.up.1.block.0.norm2.bias']
                    
    decoder_weights['15.conv1.weight']     = org_weights['first_stage_model.decoder.up.1.block.0.conv1.weight']
    decoder_weights['15.conv1.bias']       = org_weights['first_stage_model.decoder.up.1.block.0.conv1.bias']
    decoder_weights['15.conv2.weight']     = org_weights['first_stage_model.decoder.up.1.block.0.conv2.weight']
    decoder_weights['15.conv2.bias']       = org_weights['first_stage_model.decoder.up.1.block.0.conv2.bias']
                    
    decoder_weights['15.resize.weight']    = org_weights['first_stage_model.decoder.up.1.block.0.nin_shortcut.weight']
    decoder_weights['15.resize.bias']      = org_weights['first_stage_model.decoder.up.1.block.0.nin_shortcut.bias']
                    
    decoder_weights['16.norm1.weight']    = org_weights['first_stage_model.decoder.up.1.block.1.norm1.weight']
    decoder_weights['16.norm1.bias']      = org_weights['first_stage_model.decoder.up.1.block.1.norm1.bias']
    decoder_weights['16.norm2.weight']    = org_weights['first_stage_model.decoder.up.1.block.1.norm2.weight']
    decoder_weights['16.norm2.bias']      = org_weights['first_stage_model.decoder.up.1.block.1.norm2.bias']
    decoder_weights['16.conv1.weight']    = org_weights['first_stage_model.decoder.up.1.block.1.conv1.weight']
    decoder_weights['16.conv1.bias']      = org_weights['first_stage_model.decoder.up.1.block.1.conv1.bias']
    decoder_weights['16.conv2.weight']    = org_weights['first_stage_model.decoder.up.1.block.1.conv2.weight']
    decoder_weights['16.conv2.bias']      = org_weights['first_stage_model.decoder.up.1.block.1.conv2.bias']
                    
    decoder_weights['17.norm1.weight']    = org_weights['first_stage_model.decoder.up.1.block.2.norm1.weight']
    decoder_weights['17.norm1.bias']      = org_weights['first_stage_model.decoder.up.1.block.2.norm1.bias']
    decoder_weights['17.norm2.weight']    = org_weights['first_stage_model.decoder.up.1.block.2.norm2.weight']
    decoder_weights['17.norm2.bias']      = org_weights['first_stage_model.decoder.up.1.block.2.norm2.bias']
    decoder_weights['17.conv1.weight']    = org_weights['first_stage_model.decoder.up.1.block.2.conv1.weight']
    decoder_weights['17.conv1.bias']      = org_weights['first_stage_model.decoder.up.1.block.2.conv1.bias']
    decoder_weights['17.conv2.weight']    = org_weights['first_stage_model.decoder.up.1.block.2.conv2.weight']
    decoder_weights['17.conv2.bias']      = org_weights['first_stage_model.decoder.up.1.block.2.conv2.bias']
                    
    decoder_weights['19.weight']          = org_weights['first_stage_model.decoder.up.1.upsample.conv.weight']
    decoder_weights['19.bias']            = org_weights['first_stage_model.decoder.up.1.upsample.conv.bias']
                    

    decoder_weights['20.norm1.weight']            = org_weights['first_stage_model.decoder.up.0.block.0.norm1.weight']
    decoder_weights['20.norm1.bias']              = org_weights['first_stage_model.decoder.up.0.block.0.norm1.bias']
    decoder_weights['20.norm2.weight']            = org_weights['first_stage_model.decoder.up.0.block.0.norm2.weight']
    decoder_weights['20.norm2.bias']              = org_weights['first_stage_model.decoder.up.0.block.0.norm2.bias']
    decoder_weights['20.conv1.weight']            = org_weights['first_stage_model.decoder.up.0.block.0.conv1.weight']
    decoder_weights['20.conv1.bias']              = org_weights['first_stage_model.decoder.up.0.block.0.conv1.bias']
    decoder_weights['20.conv2.weight']            = org_weights['first_stage_model.decoder.up.0.block.0.conv2.weight']
    decoder_weights['20.conv2.bias']              = org_weights['first_stage_model.decoder.up.0.block.0.conv2.bias']
                    
    decoder_weights['20.resize.weight']           = org_weights['first_stage_model.decoder.up.0.block.0.nin_shortcut.weight']
    decoder_weights['20.resize.bias']             = org_weights['first_stage_model.decoder.up.0.block.0.nin_shortcut.bias']
                    

    decoder_weights['21.norm1.weight']            = org_weights['first_stage_model.decoder.up.0.block.1.norm1.weight']
    decoder_weights['21.norm1.bias']              = org_weights['first_stage_model.decoder.up.0.block.1.norm1.bias']
    decoder_weights['21.norm2.weight']            = org_weights['first_stage_model.decoder.up.0.block.1.norm2.weight']
    decoder_weights['21.norm2.bias']              = org_weights['first_stage_model.decoder.up.0.block.1.norm2.bias']
    decoder_weights['21.conv1.weight']            = org_weights['first_stage_model.decoder.up.0.block.1.conv1.weight']
    decoder_weights['21.conv1.bias']              = org_weights['first_stage_model.decoder.up.0.block.1.conv1.bias']
    decoder_weights['21.conv2.weight']            = org_weights['first_stage_model.decoder.up.0.block.1.conv2.weight']
    decoder_weights['21.conv2.bias']              = org_weights['first_stage_model.decoder.up.0.block.1.conv2.bias']
                    
    decoder_weights['22.norm1.weight']            = org_weights['first_stage_model.decoder.up.0.block.2.norm1.weight']
    decoder_weights['22.norm1.bias']              = org_weights['first_stage_model.decoder.up.0.block.2.norm1.bias']
    decoder_weights['22.norm2.weight']            = org_weights['first_stage_model.decoder.up.0.block.2.norm2.weight']
    decoder_weights['22.norm2.bias']              = org_weights['first_stage_model.decoder.up.0.block.2.norm2.bias']
    decoder_weights['22.conv1.weight']            = org_weights['first_stage_model.decoder.up.0.block.2.conv1.weight']
    decoder_weights['22.conv1.bias']              = org_weights['first_stage_model.decoder.up.0.block.2.conv1.bias']
    decoder_weights['22.conv2.weight']            = org_weights['first_stage_model.decoder.up.0.block.2.conv2.weight']
    decoder_weights['22.conv2.bias']              = org_weights['first_stage_model.decoder.up.0.block.2.conv2.bias']
                    

    decoder_weights['23.weight']          = org_weights['first_stage_model.decoder.norm_out.weight']
    decoder_weights['23.bias']            = org_weights['first_stage_model.decoder.norm_out.bias']
    decoder_weights['25.weight']          = org_weights['first_stage_model.decoder.conv_out.weight']
    decoder_weights['25.bias']            = org_weights['first_stage_model.decoder.conv_out.bias']


                    


    decoder_weights['3.attn.w_qkv.weight'] = torch.cat([org_weights['first_stage_model.decoder.mid.attn_1.q.weight'], 
                                                        org_weights['first_stage_model.decoder.mid.attn_1.k.weight'],
                                                        org_weights['first_stage_model.decoder.mid.attn_1.v.weight']], dim=0).squeeze()
    decoder_weights['3.attn.w_qkv.bias'] = torch.cat([org_weights['first_stage_model.decoder.mid.attn_1.q.bias'],
                                                      org_weights['first_stage_model.decoder.mid.attn_1.k.bias'],
                                                      org_weights['first_stage_model.decoder.mid.attn_1.v.bias']], dim=0)

    return decoder_weights


