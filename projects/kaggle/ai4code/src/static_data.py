#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   static_data.py
#        \author   chenghuige  
#          \date   2022-07-15 10:57:20.721105
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 

#sampled fold 0 id for online test purpose
SAMPLE_IDS = set(['60de206b6ce82a', '63f08d7ba7e67d', '26fc66ff507979',
       'eac0f44510c940', 'b08618d68d8944', '5a43e808cd4eeb',
       'fbc01212540d96', 'c3f79b88177999', '50b373eff030e9',
       'fb493afc9137f2', '1756fd2f2bcaeb', '5ca06a49134e94',
       'b5bbeb565abdb2', '6e65077d95d291', '072988fea7f24f',
       '89043fc749b454', 'fc801d9483c98a', '9b16410a01bc18',
       'ae5183a93c12f5', '7c74be67b7cb26', '8b2de0840b86f3',
       '8a7c3e73a0ea8f', 'ae5aebd1d2bdb6', 'fa185cbcafee21',
       '6764a50c754b41', '95afa173a1bada', 'd58edc1ff4868b',
       'a4940a5f2bdec8', 'ba7cd6485d0a30', '66d3a933504cfe',
       '94b3f452bde56a', '7813881d2cac01', '4766c2a18ec9f6',
       'b654321ad2729c', 'afead6eaec311e', 'ef5defb001c002',
       'dc0e2a79cdf89c', '9bc00ac955adc0', '797f683c0b51f3',
       '77a00e97dc9625', '1f2aecf52bd7fa', '12bb5c8ce06644',
       'dd8b72fdb2c201', '88a21b32f1fe78', '061a42cc62fd25',
       '9bdfa8da0dfd36', 'c684ce09fb95ac', '0ab18728d1ad59',
       '08cf0736eb80f3', '2d32b27bd97408', '59ab0e0f9279b9',
       'c1a12cfa9bd53a', 'd633ba699593b1', '9cf418cf9ab7e0',
       '81e5781ecbd4d0', '202c3b87bb813a', '97c553aa61a067',
       '3078b2da85cf37', '638c9121aef256', '61d15258600b37',
       '7d39716313d432', '1a50a74732d238', '31f8d7dbc2124b',
       '5c34f9d9e389f5', 'ba20ea6ad1c83e', 'f25c9a90f5b296',
       '90f71949b1a6ad', 'ef5114870ab4a7', '83da946b360f82',
       '8d4e876728690b', 'ce14ee8dd62c17', 'b7381fd0bd28ea',
       '87c0b2dcc0c36c', '4b057f5e5adfb9', '5a61454af2e54c',
       'd14fbd648b370b', '8b080fa7ed2336', '65d2c72a3f3b08',
       'c5273ad86d022b', '75de630c3fdc79', 'c47ba544d66143',
       'af9bad483f43dc', 'f014230c424a89', 'cc077a20d54b13',
       '46594e54f59bc4', '30375ddada0002', 'f1900902a286d3',
       '2b5276c0d005e0', '4e47a93c25becd', 'bd119f31ca3949',
       'b0521bdc6d6e05', 'fe5974ac705ec2', '20273bf81e0426',
       '5b3088a85f11b1', 'a86b559fac70b5', '68075cab273806',
       'bee3326b176277', '5e8cbd961ec3b4', '2f08709992fae2',
       'd3c93dc9698379', '161ed6e5eb2928', '82412c540f7d12',
       '77f8454b56a1b1', '5d0c47ac85fbf4', 'ff0b08cde8b08f',
       'bc605b932e4928', '1274cf0fef9383', '5f13f44470bd4e',
       '81751813372fbf', 'cb563c9779a44a', 'c45ee468d8d062',
       'ad937ed482a016', '4d8c2858133670', 'c120e3aba8da4b',
       'a46e9c0bd77b52', '46bff4a7906514', '467bbcabaf5cec',
       '81fbd37dbb5a5b', 'b7c407e1e3cbde', '1d2917f3382d27',
       'a239d4073b7100', '9751d0438a9040', '0700aee042661b',
       '6b4a1929446897', 'decec4e835e4e5', '1ce21386eb001b',
       'b37dbff9dbc5ba', '57d8903bb393e3', '3863cd6cf4a506',
       '69d61e2c745adc', '05bcdd224d9c16', '4b14696feece32',
       'e2b3cd93c5f59c', 'c3eb070d615f94', 'd285836b082d9e',
       '850820f0051638', 'a5652de2a3a694', 'eecf594aff41e6',
       '9eb086d6323a9c', '51d3da3ffcd4e1', '556a2f1a05a349',
       'f9c3758212a86a', '6e4070093858b4', '0960b13787b4d3',
       '8ae4426aee5699', 'f492352e110bb5', '34a06fd62a8338',
       '8e1d99189f511b', '217c8a149a613b', '88419f5917da06',
       '99384c04b0f618', '1bebd3f843c0c1', '01920f87af0b35',
       '104c31747ed7a0', '06aaa94afd514b', '0a62ae95f2377f',
       '7f0fbdee67ac81', '40547799f3e2a2', '5e4bc0dbef16b9',
       '597a5054b11435', '0c07c6efc22782', 'bde44cbdd234cb',
       '3ed5df6f9c48fc', 'd8cfe21016b5ca', 'a1c2571e304f44',
       '5ef3366e6055e2', '67c4fd0ed98294', '9553d05c9570a0',
       '893a32d39b9846', '3a16ad0a599a5d', '307e8ea940dee5',
       'dc0a5f14655235', 'c5d78b51854694', '41ce1750a0c539',
       '4f886e0fee67f2', '05be94d42bcdcc', '68902c8b64fc5b',
       '031f7da4e34886', '7fb701bd523a4e', '7b729b2812f1d2',
       '386613f3098617', '227426d689f452', 'fccc51d152d571',
       '633e0541fff51f', '7bbee4b52dea01', 'a22c7d459f2464',
       'b8db160544313e', '70e3a5cc107929', 'c6569fd069cede',
       'fe37b19e83dc98', '1ee6f101c240af', '67823c3eecf830',
       '6ac6b6b1a72e26', '98b885f8b5bfef', 'e96744235e776a',
       '528c45fe0fecc2', 'fcd9cd0de58d37', '63bb5baf863624',
       'd88639d3f3d0b6', '746c5dbe8415c8', '55a5df2427cfd1',
       '521ecbcf06adc8', '8e1c230b430f82', '4f31d037213d68',
       '51b963346aab78', '3ba34bf4b20bc1', '0f6258fd149484',
       '4a5c700e4692dd', 'ec8083b9fa75a7', '19caaf8e77c10b',
       'ad91f84a30d0a3', '32e5836b558f98', 'c569082da3fb9d',
       'a791e60ce51143', '6298ba528b5255', 'f9619b6dec2303',
       '52bb416ba7f176', '676f55aa1852a3', '6da83e6e680802',
       'c7143105c981e8', '5b72f82c9d0926', '6dd115abe2a703',
       '0a896ab4b874a6', 'b14d160cd05e67', 'f015d0147e8fbf',
       'd45666c4fb9cfe', '0490603b0ba8a7', 'a4e138f4473e21',
       '42c75bfb827883', '5fa665be44a2ef', '06e48e16117c82',
       '1bb9361e06db40', 'a1ea54c1768989', '699934c7f72114',
       '2d543a43ef8312', 'a6ccbe0295de6c', '8d82d3a82c9350',
       '386ffbefc676cf', '69d54858a8edc8', 'abf71afca425b1',
       'a060712b49beca', '237000d277f550', '4f6509ca44aeca',
       'fb98b23426c038', '8c4c201a7cd876', '1318ce8ba424f8',
       'e210cc98ab5b7d', '6a17d813eba1d4', '3d194c9e6ca5b9',
       '1eb801aeb223db', '2b7c45808ae183', '777a6e3b927b5b',
       'df53b725dfceca', '80c094360385bf', '12c6abe4998606',
       '3379e2966987ba', 'b8f871785c0a4d', 'b83b57924722b6',
       'f4f13b00668c4c', '24d9b0ce164483', '999a76f5dfd52e',
       'e52d7bd5c244c5', '8f2c9ccea5044a', '25666a01847e0c',
       '179ec0fbf6bb6a', '2761b59d4baf48', '27882eb85f9120',
       '7d2ae6e6302026', '6251bf3ad67db2', '89d172ca5787b8',
       '8c124cd0220cf7', 'a2233f43c8b209', '2641a787d1ed8b',
       '94d4d431393aa5', '0bb9f52012fff8', '8536dbe21636c3',
       'b2e392eacaae68', '50c04871ef23b4', 'da965e344ef3e6',
       'cc08fe584ce97a', '1453654a3cbfb4', 'f3953981e48154',
       'c397f18b2100e9', 'aecbaaf385edce', '1973aa72d366d4',
       'd0e06a48fd3586', '49120a8d970b01', '55147acbe1bdb9',
       '8de23fbd0ab55a', 'f40d07159a39b1', '5b65642ebfff18',
       '9876666c632414', '722a914975ea52', '05cacfec9ae79f',
       '9dade5266bcb8a', '0742bbbc47cc1d', '3ff315ef2c015f',
       'f8228c764d4a83', '7468d6e10fafd8', '09e285ca28b7d2',
       '899a9ddc24623f', '79d3894dfa09e6', 'f7a4a588d935ad',
       '8b6bff2bcc9882', '6c39e23b052257', '491b5ece382aa4',
       'ab56d5c66c05f6', '28e93a34f3b152', 'd807357c560745',
       '5a5fb703fd94fe', '80e29cd0113fe7', '8930ae48f94b3a',
       'b0022bfc28068a', 'd69d776d9aa280', '3787a918371759',
       '8ec39376d987a3', '292207e3bb9bc5', '87a1fdbadf73c0',
       'b29bad890eadde', 'e5a87918f27268', '639b77720a19c4',
       'ffdd6d6ae5b35a', 'e91db78d420eac', '5c884c77130c59',
       '7e9fbf72a03e72', '683f8cc5ff1b1e', 'ae23dcb905b6bf',
       'b56b0883265047', '3cfb466967d5ec', '65a5a53ae7632a',
       '058c6857252b82', 'ccc304648330c2'])