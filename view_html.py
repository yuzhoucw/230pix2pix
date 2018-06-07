print("<table>")
# print("<tr><td> </td>\
# <td>source</td>\
# <td>unet default</td>\
# <td>unet instance</td>\
# <td>res6 default</td>\
# <td>res6 instance</td>\
# <td>res6 kaiming</td>\
# <td>gt</td>\
# </tr>")
for i in range(1, 1099):
    print("<tr><td> </td>\
    <td>source</td>\
    <td>norm none</td>\
    <td>imageGAN</td>\
    <td>d_freq = 2</td>\
    <td>lambd = 20</td>\
    <td>lr = 2e-5</td>\
    <td>gt</td>\
    </tr>")

    print("<tr><td>{}</td>\
    <td><img width=256 height=256 src='test/testA/{}_A.jpg' /></td> \
    <td><img src='test/norm_none/test_{}.png' /></td> \
    <td><img src='test/d286/test_{}.png' /> \
    <td><img src='test/d_freq_2/test_{}.png' /> \
    <td><img src='test/lambd_20/test_{}.png' /> \
    <td><img src='test/lr_2e-5/test_{}.png' /> \
    <td><img width=256 height=256 src='test/testB/{}_B.jpg' /></tr>".format(i, i, i, i, i, i, i, i))
print("</table>")
