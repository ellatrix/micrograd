module Jekyll
    module ReplaceScriptWithPreFilter
        @@script_content = {}
        def replace_script_with_pre(input)
            # Get the current page's filename
            current_page = @context.registers[:page]['path']
            current_filename = File.basename(current_page, File.extname(current_page))

            input.gsub( /<script(.*?)>(.*?)<\/script>/m ) do
                attributes = $1
                content = $2
                if attributes =~ /data-src=["'](.*?)["']/
                    data_src = $1
                    # Accumulate script content keyed by data src
                    @@script_content["#{current_filename}-#{data_src}"] ||= ""
                    @@script_content["#{current_filename}-#{data_src}"] << "#{content}"
                end
                num_newlines = content.scan("\n").count
                "<textarea disabled rows='#{num_newlines}'#{attributes}>#{content}</textarea>"
            end
        end
        def self.write_script_files
            @@script_content.each do |data_src, content|
                output_file = "_site/#{data_src}"
                File.write(output_file, content)
                puts "Script content written to #{output_file}"
            end
            @@script_content.clear
        end
    end
end

Liquid::Template.register_filter(Jekyll::ReplaceScriptWithPreFilter)

Jekyll::Hooks.register :site, :post_write do |site|
    Jekyll::ReplaceScriptWithPreFilter.write_script_files
end
