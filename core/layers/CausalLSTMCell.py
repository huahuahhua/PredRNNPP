import torch
import torch.nn as nn


class CausalLSTMCell(nn.Module):
    def __init__(self, in_channel, hidden_dim, configs):
        super(CausalLSTMCell, self).__init__()
        self.batch = configs.batch_size
        self.input_channels = in_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = configs.filter_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.device = configs.device
        self.width = configs.img_width // configs.patch_size // configs.sr_size
        self.height = configs.img_height // configs.patch_size // configs.sr_size
        self._forget_bias = 1.0

        if configs.layer_norm:
            self.conv_h_cc = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 4, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim * 4, self.width, self.height]))
            self.conv_c_cc = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 3, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim * 3, self.width, self.height]))
            self.conv_m_cc = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 3, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim * 3, self.width, self.height]))
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(self.input_channels, self.hidden_dim * 7, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim * 7, self.width, self.height]))
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 4, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim * 4, self.width, self.height]))
            self.conv_o_m = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim, self.width, self.height]))
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim, self.width, self.height]))
            self.conv_cell = nn.Sequential(
                nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False),
                nn.LayerNorm([self.hidden_dim, self.width, self.height]))
        else:
            self.conv_h_cc = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 4, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_c_cc = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 3, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_m_cc = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 3, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(self.input_channels, self.hidden_dim * 7, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 4, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_o_m = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))
            self.conv_cell = nn.Sequential(
                nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=False))

    def forward(self, x, h, c, m):
        # if h is None:
        #     h = self.init_state()
        # if c is None:
        #     c = self.init_state()
        if m is None:
            m = self.init_state()

        h_cc = self.conv_h_cc(h)
        c_cc = self.conv_c_cc(c)
        m_cc = self.conv_m_cc(m)

        i_h, g_h, f_h, o_h = torch.split(h_cc, self.hidden_dim, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.hidden_dim, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.hidden_dim, 1)
        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.conv_x_cc(x)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc, self.hidden_dim, 1)
            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)
        c_new = f * c + i * g
        c2m = self.conv_c2m(c_new)

        i_c, g_c, f_c, o_c = torch.split(c2m, self.hidden_dim, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)
        m_new = ff * torch.tanh(m_m) + ii * gg
        o_m = self.conv_o_m(m_new)

        if x is None:
            o = torch.tanh(o_c + o_m + o_h)

        else:
            o = torch.tanh(o_x + o_c + o_m + o_h)

        cell = torch.cat([c_new, m_new], 1)
        cell = self.conv_cell(cell)

        h_new = o * torch.tanh(cell)

        return h_new, c_new, m_new

    def init_state(self):
        return torch.zeros((self.batch, self.hidden_dim, self.height, self.width), dtype=torch.float32).to(self.device)


class GHU(nn.Module):
    def __init__(self, in_channel, hidden_dim, configs):
        super(GHU, self).__init__()
        """Initialize the Gradient Highway Unit.
        """
        self.batch = configs.batch_size
        self.hidden_dim = in_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = configs.filter_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.device = configs.device
        self.width = configs.img_width // configs.patch_size // configs.sr_size
        self.height = configs.img_height // configs.patch_size // configs.sr_size
        self._forget_bias = 1.0
        self.device = configs.device

        if configs.layer_norm:
            self.z_concat_conv = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=self.kernel_size, padding=2,
                          bias=False),
                nn.LayerNorm([self.hidden_dim * 2, self.width, self.width])
            )
            self.x_concat_conv = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=self.kernel_size, padding=2,
                          bias=False),
                nn.LayerNorm([self.hidden_dim * 2, self.width, self.width])
            )
        else:
            self.z_concat_conv = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=self.kernel_size, padding=2, bias=False)
            )
            self.x_concat_conv = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=self.kernel_size, padding=2, bias=False)
            )

    def init_state(self):
        return torch.zeros((self.batch, self.hidden_dim, self.height, self.width), dtype=torch.float32).to(self.device)

    def forward(self, x, z):
        if z is None:
            z = self.init_state()
        z_concat = self.z_concat_conv(z)
        x_concat = self.x_concat_conv(x)

        gates = torch.add(x_concat, z_concat)

        p, u = torch.split(gates, self.hidden_dim, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new


if __name__ == '__main__':
    from configs.predrnnCausal_configs import configs

    parse = configs()
    configs = parse.parse_args()
    hidden = []

    # model = CausalLSTMCell(configs.img_channel, configs.num_hidden, configs).cuda()
    # a = torch.randn(
    #     (configs.batch_size, configs.img_channel, configs.img_width // configs.patch_size // configs.sr_size,
    #      configs.img_height // configs.patch_size // configs.sr_size
    #      )).to(configs.device)
    # h = torch.randn((configs.batch_size, configs.num_hidden, configs.img_width // configs.patch_size // configs.sr_size,
    #                  configs.img_height // configs.patch_size // configs.sr_size
    #                  )).cuda()
    # c = m = h
    # h, c, m = model(a, h, c, m)
    # print(h.shape)

    ghu = GHU(configs.num_hidden, configs.num_hidden, configs).cuda()
    data = torch.randn(
        (configs.batch_size, configs.num_hidden, configs.img_width // configs.patch_size // configs.sr_size,
         configs.img_height // configs.patch_size // configs.sr_size
         )).to(configs.device)

    z = torch.randn((configs.batch_size, configs.num_hidden, configs.img_width // configs.patch_size // configs.sr_size,
                     configs.img_height // configs.patch_size // configs.sr_size
                     )).cuda()
    data = ghu(data, z)

    print(data.shape)
