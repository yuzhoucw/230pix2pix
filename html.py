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
    <td>unet default</td>\
    <td>unet instance</td>\
    <td>res6 default</td>\
    <td>res6 instance</td>\
    <td>res6 kaiming</td>\
    <td>gt</td>\
    </tr>")

    print("<tr><td>{}</td>\
    <td><img width=256 height=256 src='images/testA/{}_A.jpg' /></td> \
    <td><img src='images/unet_default/test_{}.png' /></td> \
    <td><img src='images/unet_instance/test_{}.png' /> \
    <td><img src='images/res6_default/test_{}.png' /> \
    <td><img src='images/res6_instance/test_{}.png' /> \
    <td><img src='images/res6_kaiming/test_{}.png' /> \
    <td><img width=256 height=256 src='images/testB/{}_B.jpg' /></tr>".format(i, i, i, i, i, i, i, i))
print("</table>")